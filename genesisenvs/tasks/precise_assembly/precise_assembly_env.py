"""Genesis-based Stage-2 Precise-Assembly environment for Play2Perfect.

This subclass of :class:`GenesisPlayEnv` configures the scene for a specific
contact-rich assembly task (peg-in-hole, beam assembly, or screw insertion).
It overrides goal sampling, reward gating, and termination to match the
assembly task semantics from the Isaac Lab reference implementation.

Supported ``problem`` keys (matching ``evaluation/problems/``)::

    tight_insertion      – L-shaped peg into 0.5 mm tolerance hole
    beam_assembly_step1  – Fabrica beam part 0 into fixture
    beam_assembly_step2  – Fabrica beam part 2 into fixture
    screwing             – Furniture-bench table leg screwing

Usage::

    from genesisenvs.tasks.precise_assembly import GenesisPreciseAssemblyEnv

    env = GenesisPreciseAssemblyEnv(
        num_envs=1024,
        problem="tight_insertion",
        checkpoint=None,  # load an existing play policy for feature extraction
    )
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch

import genesis as gs

from genesisenvs.tasks.play.math_utils import (
    keypoints_world,
    quat_from_angle_axis,
    quat_mul,
    random_orientation,
)
from genesisenvs.tasks.play.play_env import GenesisPlayEnv

# ---------------------------------------------------------------------------
# Per-problem asset and goal configuration
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[4]

PROBLEM_CONFIGS: dict[str, dict] = {
    "tight_insertion": {
        "object_urdf": "assets/urdf/peg_in_hole/peg/peg.urdf",
        "fixture_urdf": "assets/urdf/peg_in_hole/holes/hole_tol0p5mm/hole_tol0p5mm.urdf",
        "object_scale": (6.25, 0.75, 0.5),
        # Pre-insert pose (local, relative to env origin): [x, y, z, qw, qx, qy, qz]
        "pre_insert_pos": (0.0, 0.0, 0.70),
        "pre_insert_quat": (1.0, 0.0, 0.0, 0.0),
        # Final insertion pose.
        "insert_pos": (0.0, 0.0, 0.60),
        "insert_quat": (1.0, 0.0, 0.0, 0.0),
        # Tighter tolerance for the assembly task.
        "insertion_success_tolerance": 0.01,
        "hole_x_range": (-0.1875, 0.1875),
        "hole_y_range": (-0.1, 0.1),
    },
    "beam_assembly_step1": {
        "object_urdf": "assets/urdf/fabrica/beam_3x/0/beam_3x_0.urdf",
        "fixture_urdf": "assets/urdf/fabrica/beam_3x/insertion_fixtures/part_0.urdf",
        "object_scale": (1.0, 1.0, 1.0),
        "pre_insert_pos": (0.0, 0.0, 0.75),
        "pre_insert_quat": (1.0, 0.0, 0.0, 0.0),
        "insert_pos": (0.0, 0.0, 0.60),
        "insert_quat": (1.0, 0.0, 0.0, 0.0),
        "insertion_success_tolerance": 0.01,
        "hole_x_range": (-0.15, 0.15),
        "hole_y_range": (-0.10, 0.10),
    },
    "beam_assembly_step2": {
        "object_urdf": "assets/urdf/fabrica/beam_3x/2/beam_3x_2.urdf",
        "fixture_urdf": "assets/urdf/fabrica/beam_3x/insertion_fixtures/part_2.urdf",
        "object_scale": (1.0, 1.0, 1.0),
        "pre_insert_pos": (0.0, 0.0, 0.75),
        "pre_insert_quat": (1.0, 0.0, 0.0, 0.0),
        "insert_pos": (0.0, 0.0, 0.60),
        "insert_quat": (1.0, 0.0, 0.0, 0.0),
        "insertion_success_tolerance": 0.01,
        "hole_x_range": (-0.15, 0.15),
        "hole_y_range": (-0.10, 0.10),
    },
    "screwing": {
        "object_urdf": "assets/urdf/furniture_bench/square_table/square_table_leg4/square_table_leg4.urdf",
        "fixture_urdf": "assets/urdf/furniture_bench/square_table/insertion_fixtures/one_leg.urdf",
        "object_scale": (1.0, 1.0, 1.0),
        "pre_insert_pos": (0.0, 0.0, 0.75),
        "pre_insert_quat": (1.0, 0.0, 0.0, 0.0),
        "insert_pos": (0.0, 0.0, 0.60),
        "insert_quat": (1.0, 0.0, 0.0, 0.0),
        "insertion_success_tolerance": 0.01,
        "hole_x_range": (-0.10, 0.10),
        "hole_y_range": (-0.10, 0.10),
    },
}

# Goal modes for the precise-assembly task.
GOAL_MODE_PRE_INSERT_AND_FINAL = "preInsertAndFinal"
GOAL_MODE_FINAL_ONLY = "finalGoalOnly"


class GenesisPreciseAssemblyEnv(GenesisPlayEnv):
    """Stage-2 Genesis environment for contact-rich assembly fine-tuning.

    Inherits all physics, observation, and training infrastructure from
    :class:`GenesisPlayEnv` and overrides only the asset loading,
    goal sampling, and success criterion to implement the assembly task.

    Parameters
    ----------
    num_envs:
        Number of parallel environments.
    problem:
        Assembly task name (one of ``PROBLEM_CONFIGS``).
    cfg:
        Optional dict of overrides merged on top of the base defaults.
    show_viewer:
        Open the Genesis interactive viewer (headful mode).
    goal_mode:
        ``"preInsertAndFinal"`` (default) provides a two-stage goal sequence:
        the policy first reaches a pre-insertion waypoint, then the final
        insertion target.  ``"finalGoalOnly"`` skips the pre-insert stage.
    hole_yaw_range_deg:
        Per-episode yaw randomisation applied to the fixture placement.
    """

    def __init__(
        self,
        num_envs: int,
        problem: str = "tight_insertion",
        cfg: Optional[dict] = None,
        show_viewer: bool = False,
        goal_mode: str = GOAL_MODE_PRE_INSERT_AND_FINAL,
        hole_yaw_range_deg: float = 0.0,
    ) -> None:
        if problem not in PROBLEM_CONFIGS:
            raise ValueError(
                f"Unknown problem {problem!r}. "
                f"Valid options: {list(PROBLEM_CONFIGS.keys())}"
            )
        self._problem_name = problem
        self._problem_cfg = PROBLEM_CONFIGS[problem]
        self._goal_mode = goal_mode
        self._hole_yaw_range_deg = hole_yaw_range_deg

        # Merge problem-level overrides into base cfg before building the scene.
        merged_cfg: dict = {
            "object_urdf": self._problem_cfg["object_urdf"],
            "object_scale": self._problem_cfg["object_scale"],
            "success_tolerance": float(
                self._problem_cfg["insertion_success_tolerance"]
            ),
            "target_success_tolerance": float(
                self._problem_cfg["insertion_success_tolerance"]
            ),
            # Assembly tasks generally need longer episodes for fine contact.
            "episode_length_s": 15.0,
            # Disable delta-goal during assembly (goal is fixed to insert target).
            "goal_sampling_type": "absolute",
        }
        merged_cfg.update(cfg or {})

        super().__init__(num_envs=num_envs, cfg=merged_cfg, show_viewer=show_viewer)

        # Fixture (hole / receptacle) state tensors.
        self._fixture_pos = torch.zeros(num_envs, 3, device=self.device)
        self._fixture_quat = torch.zeros(num_envs, 4, device=self.device)
        self._fixture_quat[:, 0] = 1.0

        # Per-env goal-stage index: 0 = pre-insert, 1 = final insert.
        # Only used when goal_mode == GOAL_MODE_PRE_INSERT_AND_FINAL.
        self._goal_stage = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # Retract state for the optional pull-away bonus.
        self._in_retract = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------
    # Override: add fixture entity before scene.build()
    # ------------------------------------------------------------------

    def _add_extra_entities(self) -> None:
        """Add the assembly fixture (hole / receptacle) to the scene."""
        fixture_urdf = str(_REPO_ROOT / self._problem_cfg["fixture_urdf"])
        self.fixture = self.scene.add_entity(
            gs.morphs.URDF(
                file=fixture_urdf,
                fixed=True,          # fixture is bolted to the table
                pos=(0.0, 0.0, float(self.cfg["table_reset_z"])),
                quat=(1.0, 0.0, 0.0, 0.0),
            )
        )

    # ------------------------------------------------------------------
    # Override: goal sampling is assembly-specific
    # ------------------------------------------------------------------

    def _reset_goal_pose(self, env_ids: torch.Tensor) -> None:
        """Set goal pose to pre-insert or final-insert target."""
        n = env_ids.numel()
        dev = self.device
        pcfg = self._problem_cfg

        # Randomise fixture placement (XY + optional yaw).
        hole_x_lo, hole_x_hi = pcfg["hole_x_range"]
        hole_y_lo, hole_y_hi = pcfg["hole_y_range"]
        fx = torch.empty(n, device=dev).uniform_(hole_x_lo, hole_x_hi)
        fy = torch.empty(n, device=dev).uniform_(hole_y_lo, hole_y_hi)
        fz = torch.full((n,), float(self.cfg["table_reset_z"]), device=dev)
        fixture_pos = torch.stack([fx, fy, fz], dim=-1)

        if self._hole_yaw_range_deg > 0.0:
            yaw = (
                torch.empty(n, device=dev).uniform_(-1.0, 1.0)
                * self._hole_yaw_range_deg * (math.pi / 180.0)
            )
            half = yaw * 0.5
            fixture_quat = torch.stack(
                [torch.cos(half),
                 torch.zeros_like(half),
                 torch.zeros_like(half),
                 torch.sin(half)],
                dim=-1,
            )
        else:
            fixture_quat = torch.zeros(n, 4, device=dev)
            fixture_quat[:, 0] = 1.0

        # Move fixture entity in the sim.
        self.fixture.set_pos(fixture_pos, envs_idx=env_ids)
        self.fixture.set_quat(fixture_quat, envs_idx=env_ids)
        self._fixture_pos[env_ids] = fixture_pos
        self._fixture_quat[env_ids] = fixture_quat

        # Set goal stage and goal pose.
        self._goal_stage[env_ids] = 0  # start from pre-insert
        self._apply_goal_for_stage(env_ids)

    def _apply_goal_for_stage(self, env_ids: torch.Tensor) -> None:
        """Write goal pos/quat for the current stage of each env in env_ids."""
        dev = self.device
        pcfg = self._problem_cfg
        n = env_ids.numel()

        stage = self._goal_stage[env_ids]  # (n,)
        pre_pos = torch.tensor(pcfg["pre_insert_pos"], device=dev)
        pre_quat = torch.tensor(pcfg["pre_insert_quat"], device=dev)
        ins_pos = torch.tensor(pcfg["insert_pos"], device=dev)
        ins_quat = torch.tensor(pcfg["insert_quat"], device=dev)

        # Select goal based on stage (0 = pre-insert, 1 = final).
        is_final = (stage == 1).float().unsqueeze(-1)
        goal_pos = (
            is_final * ins_pos.unsqueeze(0)
            + (1.0 - is_final) * pre_pos.unsqueeze(0)
        )
        goal_quat = (
            is_final * ins_quat.unsqueeze(0)
            + (1.0 - is_final) * pre_quat.unsqueeze(0)
        )
        # Translate goal into the fixture's local frame.
        from genesisenvs.tasks.play.math_utils import quat_apply
        goal_pos_world = self._fixture_pos[env_ids] + quat_apply(
            self._fixture_quat[env_ids],
            goal_pos - torch.tensor(pcfg["insert_pos"], device=dev).unsqueeze(0)
            + torch.tensor(pcfg["insert_pos"], device=dev).unsqueeze(0),
        )
        self._goal_pos[env_ids] = goal_pos_world
        self._goal_quat[env_ids] = goal_quat

    # ------------------------------------------------------------------
    # Override: advancement from pre-insert to final-insert stage
    # ------------------------------------------------------------------

    def _compute_terminations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance goal stage on pre-insert success; terminate on final insert."""
        # Check if any env in pre-insert stage has reached the pre-insert goal.
        pre_insert_done = self._is_success & (self._goal_stage == 0)
        if pre_insert_done.any() and self._goal_mode == GOAL_MODE_PRE_INSERT_AND_FINAL:
            advance_ids = pre_insert_done.nonzero(as_tuple=False).flatten()
            self._goal_stage[advance_ids] = 1
            self._apply_goal_for_stage(advance_ids)
            # Reset trackers so the new goal can be measured fresh.
            self._closest_keypoint_max_dist[advance_ids] = -1.0
            self._near_goal_steps[advance_ids] = 0

        return super()._compute_terminations()
