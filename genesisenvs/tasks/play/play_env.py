"""Genesis-based Stage-1 Play environment for Play2Perfect.

This module re-implements the Isaac Lab ``PlayEnv`` using the Genesis physics
simulator so that the full Play2Perfect two-stage pipeline can be trained
without an Isaac Sim licence.

Design goals
------------
* Match the RSL-RL ``OnPolicyRunner`` interface (same as the pilla_rl
  reference) so the same training script works with any future simulator.
* Faithfully port all reward, termination, and observation logic from the
  ``isaacsimenvs`` reference implementation.
* Use only Genesis and pure-PyTorch — no Isaac Lab / Isaac Sim imports.

Interface contract (rsl-rl-lib 2.3.3)
--------------------------------------
``step(actions)``         → ``(obs, rew, done, extras)``
``get_observations()``    → ``(obs, extras)``
``get_privileged_observations()`` → ``None`` (privileged obs live in extras)
``reset()``               → ``(obs, extras)``
Attributes: ``num_envs``, ``num_obs``, ``num_privileged_obs``, ``num_actions``
``extras["time_outs"]``                  – (N,) float, 1 when time limit hit
``extras["observations"]["critic"]``     – privileged state observations

Quaternion convention: **wxyz** throughout (matching Genesis and Isaac Lab).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

import genesis as gs
from genesis.utils.geom import inv_quat, transform_by_quat

from genesisenvs.tasks.play.math_utils import (
    KEYPOINT_CORNERS,
    keypoints_world,
    perturb_quat,
    quat_apply,
    quat_from_angle_axis,
    quat_mul,
    random_orientation,
)

# ---------------------------------------------------------------------------
# Repository root — used to resolve URDF paths at init time.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Robot joint layout (29 DOF: 7 arm + 22 hand)
# Canonical order that the policy was trained against in the original Isaac Lab
# implementation.  Genesis DOF indices are resolved from joint names at init.
# ---------------------------------------------------------------------------

JOINT_NAMES_CANONICAL: tuple[str, ...] = (
    # KUKA iiwa14 arm (7)
    "iiwa14_joint_1", "iiwa14_joint_2", "iiwa14_joint_3", "iiwa14_joint_4",
    "iiwa14_joint_5", "iiwa14_joint_6", "iiwa14_joint_7",
    # Sharpa dexterous hand — thumb (5)
    "left_1_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE",
    "left_thumb_MCP_AA", "left_thumb_IP",
    # index (4)
    "left_2_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP", "left_index_DIP",
    # middle (4)
    "left_3_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP", "left_middle_DIP",
    # ring (4)
    "left_4_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP", "left_ring_DIP",
    # pinky (5)
    "left_5_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA",
    "left_pinky_PIP", "left_pinky_DIP",
)
assert len(JOINT_NAMES_CANONICAL) == 29, "Expected 29 canonical joint names"

NUM_ARM_JOINTS: int = 7
NUM_HAND_JOINTS: int = 22
NUM_JOINTS: int = NUM_ARM_JOINTS + NUM_HAND_JOINTS

PALM_LINK_NAME: str = "iiwa14_link_7"
FINGERTIP_LINK_NAMES: tuple[str, ...] = (
    "left_index_DP", "left_middle_DP", "left_ring_DP",
    "left_thumb_DP", "left_pinky_DP",
)
NUM_FINGERTIPS: int = len(FINGERTIP_LINK_NAMES)

# Small offset from the palm link origin to the approximate palm centre.
PALM_CENTER_OFFSET: tuple[float, float, float] = (-0.0, -0.02, 0.16)
FINGERTIP_OFFSET: tuple[float, float, float] = (0.02, 0.002, 0.0)

NUM_KEYPOINTS: int = 4

# ---------------------------------------------------------------------------
# Observation field sizes — must match the Isaac Lab reference exactly so that
# checkpoints can be transferred between simulators.
# ---------------------------------------------------------------------------

OBS_FIELD_SIZES: dict[str, int] = {
    "joint_pos":               NUM_JOINTS,
    "joint_vel":               NUM_JOINTS,
    "prev_action_targets":     NUM_JOINTS,
    "palm_pos":                3,
    "palm_rot":                4,
    "palm_vel":                6,
    "object_rot":              4,
    "object_vel":              6,
    "fingertip_pos_rel_palm":  3 * NUM_FINGERTIPS,   # 15
    "keypoints_rel_palm":      3 * NUM_KEYPOINTS,    # 12
    "keypoints_rel_goal":      3 * NUM_KEYPOINTS,    # 12
    "object_scales":           3,
    "closest_keypoint_max_dist": 1,
    "closest_fingertip_dist":  NUM_FINGERTIPS,       # 5
    "lifted_object":           1,
    "progress":                1,
    "successes":               1,
    "reward":                  1,
}

# Actor (policy) obs fields — no privileged information.
ACTOR_OBS_FIELDS: tuple[str, ...] = (
    "joint_pos", "joint_vel", "prev_action_targets",
    "palm_pos", "palm_rot",
    "object_rot",
    "fingertip_pos_rel_palm",
    "keypoints_rel_palm", "keypoints_rel_goal",
    "object_scales",
)

# Critic (privileged state) obs fields — full information.
CRITIC_OBS_FIELDS: tuple[str, ...] = (
    "joint_pos", "joint_vel", "prev_action_targets",
    "palm_pos", "palm_rot", "palm_vel",
    "object_rot", "object_vel",
    "fingertip_pos_rel_palm",
    "keypoints_rel_palm", "keypoints_rel_goal",
    "object_scales",
    "closest_keypoint_max_dist",
    "closest_fingertip_dist",
    "lifted_object",
    "progress",
    "successes",
    "reward",
)


def _obs_dim(fields: tuple[str, ...]) -> int:
    return sum(OBS_FIELD_SIZES[f] for f in fields)


# ---------------------------------------------------------------------------
# Default configuration dicts (mirrors Play.yaml defaults)
# ---------------------------------------------------------------------------

DEFAULT_ENV_CFG: dict = {
    # Scene
    "num_envs": 4096,
    "dt": 1.0 / 60.0,          # 60 Hz control
    "physics_dt": 1.0 / 120.0,  # 120 Hz physics
    "episode_length_s": 10.0,
    # Robot URDF (relative to repo root)
    "robot_urdf": "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf",
    # Table URDF
    "table_urdf": "assets/urdf/table_narrow.urdf",
    # Object URDF (one type per training run for Stage 1)
    "object_urdf": "assets/urdf/handle_head_primitives/hammer/hammer_0.urdf",
    # Object default size for keypoint reward (m, half-extents × keypoint_scale × 0.5)
    "object_base_size": 0.04,
    "object_scale": (1.0, 1.0, 1.0),
    # PD gains (shared for all arm joints; hand joints use different values)
    "arm_kp": [600.0, 600.0, 500.0, 400.0, 200.0, 200.0, 200.0],
    "arm_kd": [27.03, 27.03, 24.67, 22.07, 9.75, 9.15, 9.15],
    "hand_kp": 5.0,
    "hand_kd": 0.2,
    # Action
    "action_space": NUM_JOINTS,          # 29
    "dof_speed_scale": 1.5,
    "arm_moving_average": 0.1,
    "hand_moving_average": 0.1,
    "clip_actions": 1.0,
    # Observation
    "clamp_abs_observations": 10.0,
    # Reset
    "reset_position_noise_x": 0.1,
    "reset_position_noise_y": 0.1,
    "reset_position_noise_z": 0.02,
    "reset_dof_pos_random_interval_arm": 0.1,
    "reset_dof_pos_random_interval_fingers": 0.1,
    "reset_dof_vel_random_interval": 0.5,
    "table_reset_z": 0.38,
    "table_reset_z_range": 0.01,
    "table_object_z_offset": 0.25,
    "goal_sampling_type": "delta",       # "delta" | "absolute"
    "delta_goal_distance": 0.1,
    "delta_rotation_degrees": 90.0,
    "target_volume_mins": [-0.35, -0.2, 0.6],
    "target_volume_maxs": [0.35, 0.2, 0.95],
    # Reward
    "keypoint_rew_scale": 200.0,
    "keypoint_scale": 1.5,
    "fixed_size": [0.141, 0.03025, 0.0271],
    "fixed_size_keypoint_reward": True,
    "lifting_rew_scale": 20.0,
    "lifting_bonus": 300.0,
    "lifting_bonus_threshold": 0.15,
    "distance_delta_rew_scale": 50.0,
    "reach_goal_bonus": 1000.0,
    "kuka_actions_penalty_scale": 0.03,
    "hand_actions_penalty_scale": 0.003,
    # Termination
    "success_tolerance": 0.075,
    "target_success_tolerance": 0.01,
    "success_steps": 10,
    "max_consecutive_successes": 50,
    "force_consecutive_near_goal_steps": False,
    "tolerance_curriculum_increment": 0.9,
    "tolerance_curriculum_interval": 3000,
    # Domain randomisation
    "use_obs_delay": True,
    "obs_delay_max": 3,
    "use_action_delay": True,
    "action_delay_max": 3,
    "use_object_state_delay_noise": True,
    "object_state_delay_max": 10,
    "object_state_xyz_noise_std": 0.01,
    "object_state_rotation_noise_degrees": 5.0,
    "joint_velocity_obs_noise_std": 0.1,
}


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------


class GenesisPlayEnv:
    """Genesis-backed Stage-1 play environment.

    Parameters
    ----------
    num_envs:
        Number of parallel environments.
    cfg:
        Dictionary of configuration values (defaults from ``DEFAULT_ENV_CFG``).
    show_viewer:
        Open the Genesis interactive viewer (headful mode).
    """

    def __init__(
        self,
        num_envs: int,
        cfg: Optional[dict] = None,
        show_viewer: bool = False,
    ) -> None:
        self.num_envs = num_envs
        self.cfg = {**DEFAULT_ENV_CFG, **(cfg or {})}
        c = self.cfg

        self.device = gs.device
        self.dt = float(c["dt"])
        self.physics_dt = float(c["physics_dt"])
        self.decimation = max(1, round(self.dt / self.physics_dt))
        self.max_episode_length = math.ceil(c["episode_length_s"] / self.dt)

        self.num_actions: int = NUM_JOINTS
        self.num_obs: int = _obs_dim(ACTOR_OBS_FIELDS)
        self.num_privileged_obs: int = _obs_dim(CRITIC_OBS_FIELDS)

        self._build_scene(show_viewer)
        self._resolve_dof_and_link_indices()
        self._set_pd_gains()
        self._allocate_buffers()

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _build_scene(self, show_viewer: bool) -> None:
        """Create the Genesis scene, add entities, then build.

        The build sequence is:
          1. ``_create_scene()``       — instantiate ``gs.Scene``
          2. ``_add_entities()``       — add ground / table / robot / object
          3. ``_add_extra_entities()`` — hook for subclasses (e.g. fixture)
          4. ``scene.build()``         — finalise multi-env layout

        Subclasses that need additional entities (e.g. a fixture URDF for
        assembly tasks) should override ``_add_extra_entities()`` only.
        """
        self._create_scene(show_viewer)
        self._add_entities()
        self._add_extra_entities()
        self.scene.build(n_envs=self.num_envs, env_spacing=(1.5, 1.5))

    def _create_scene(self, show_viewer: bool) -> None:
        """Instantiate the ``gs.Scene`` with physics and viewer options."""
        physics_dt = self.physics_dt
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=physics_dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(1.5, 0.0, 1.5),
                camera_lookat=(0.0, 0.0, 0.6),
                camera_fov=45,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            rigid_options=gs.options.RigidOptions(
                dt=physics_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

    def _add_entities(self) -> None:
        """Add the base scene entities: ground plane, table, robot, object."""
        c = self.cfg

        # Ground plane (safety floor beneath the table).
        self.scene.add_entity(gs.morphs.Plane(pos=(0, 0, 0), fixed=True))

        # Table — fixed rigid body.
        table_urdf = str(_REPO_ROOT / c["table_urdf"])
        self.table = self.scene.add_entity(
            gs.morphs.URDF(file=table_urdf, fixed=True, pos=(0.0, 0.0, 0.0))
        )

        # Robot — KUKA iiwa14 + Sharpa dexterous hand.  Base is fixed.
        robot_urdf = str(_REPO_ROOT / c["robot_urdf"])
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=robot_urdf,
                fixed=True,       # fixes the base link (arm is wall-mounted)
                pos=(0.0, 0.0, 0.0),
                quat=(1.0, 0.0, 0.0, 0.0),  # wxyz
            )
        )

        # Manipulated object.  Object type is selected via ``cfg["object_urdf"]``.
        object_urdf = str(_REPO_ROOT / c["object_urdf"])
        init_obj_z = float(c["table_reset_z"]) + float(c["table_object_z_offset"])
        self.object = self.scene.add_entity(
            gs.morphs.URDF(
                file=object_urdf,
                fixed=False,
                pos=(0.0, 0.0, init_obj_z),
                quat=(1.0, 0.0, 0.0, 0.0),
            )
        )

    def _add_extra_entities(self) -> None:
        """Hook for subclasses to add additional entities before ``scene.build()``.

        The base implementation is a no-op.  Override in subclasses to add
        fixtures, receptacles, or other task-specific objects.
        """

    # ------------------------------------------------------------------
    # Post-build index resolution
    # ------------------------------------------------------------------

    def _resolve_dof_and_link_indices(self) -> None:
        """Resolve DOF indices and link local indices after scene.build()."""
        # DOF indices for all 29 controlled joints (in canonical policy order).
        self._dof_idx: list[int] = [
            self.robot.get_joint(name).dof_start
            for name in JOINT_NAMES_CANONICAL
        ]
        self._arm_dof_idx: list[int] = self._dof_idx[:NUM_ARM_JOINTS]
        self._hand_dof_idx: list[int] = self._dof_idx[NUM_ARM_JOINTS:]

        # Link-local indices for palm and fingertips.
        self._palm_link_idx: int = self.robot.get_link(PALM_LINK_NAME).idx_local
        self._fingertip_link_idx: list[int] = [
            self.robot.get_link(name).idx_local
            for name in FINGERTIP_LINK_NAMES
        ]

        # Joint position limits — (2, 29) or query per-env; use env 0.
        lower, upper = self.robot.get_dofs_limit(self._dof_idx)
        self._joint_lower = lower.to(self.device)   # (29,)
        self._joint_upper = upper.to(self.device)   # (29,)

        # Normalisation denominator for joint_pos observation.
        self._joint_range = (self._joint_upper - self._joint_lower).clamp(min=1e-6)

        # Separate arm / hand limits (for action clamping).
        self._arm_lower = self._joint_lower[:NUM_ARM_JOINTS].unsqueeze(0).expand(self.num_envs, -1)
        self._arm_upper = self._joint_upper[:NUM_ARM_JOINTS].unsqueeze(0).expand(self.num_envs, -1)
        self._hand_lower = self._joint_lower[NUM_ARM_JOINTS:].unsqueeze(0).expand(self.num_envs, -1)
        self._hand_upper = self._joint_upper[NUM_ARM_JOINTS:].unsqueeze(0).expand(self.num_envs, -1)

    def _set_pd_gains(self) -> None:
        """Configure per-joint PD gains on the robot."""
        c = self.cfg
        kp_list = list(c["arm_kp"]) + [float(c["hand_kp"])] * NUM_HAND_JOINTS
        kd_list = list(c["arm_kd"]) + [float(c["hand_kd"])] * NUM_HAND_JOINTS
        self.robot.set_dofs_kp(kp_list, self._dof_idx)
        self.robot.set_dofs_kv(kd_list, self._dof_idx)

    # ------------------------------------------------------------------
    # Buffer allocation
    # ------------------------------------------------------------------

    def _allocate_buffers(self) -> None:
        """Allocate all per-env state tensors."""
        N = self.num_envs
        dev = self.device
        c = self.cfg
        f = gs.tc_float
        i = gs.tc_int

        # Episode management.
        self.episode_length_buf = torch.zeros(N, device=dev, dtype=i)
        self.reset_buf = torch.ones(N, device=dev, dtype=i)
        self.extras: dict = {"observations": {}, "time_outs": torch.zeros(N, device=dev)}

        # Robot joint state.
        self.dof_pos = torch.zeros(N, NUM_JOINTS, device=dev, dtype=f)
        self.dof_vel = torch.zeros(N, NUM_JOINTS, device=dev, dtype=f)

        # Action / target buffers.
        self._cur_targets = torch.zeros(N, NUM_JOINTS, device=dev, dtype=f)
        self._prev_targets = torch.zeros(N, NUM_JOINTS, device=dev, dtype=f)

        # Observation output buffers.
        self.obs_buf = torch.zeros(N, self.num_obs, device=dev, dtype=f)
        self.critic_obs_buf = torch.zeros(N, self.num_privileged_obs, device=dev, dtype=f)

        # Object / goal pose.
        init_z = float(c["table_reset_z"]) + float(c["table_object_z_offset"])
        self._object_init_z = torch.full((N,), init_z, device=dev)
        self._goal_pos = torch.zeros(N, 3, device=dev, dtype=f)
        self._goal_quat = torch.zeros(N, 4, device=dev, dtype=f)
        self._goal_quat[:, 0] = 1.0  # identity

        # Object scales (isotropic 1.0 for Play stage; PreciseAssembly overrides).
        scale = torch.tensor(c["object_scale"], device=dev, dtype=f)  # (3,)
        self._object_scale_per_env = scale.unsqueeze(0).expand(N, -1).contiguous()

        # Keypoint offsets (fixed-size, matching Isaac Lab default).
        corners = torch.tensor(KEYPOINT_CORNERS, device=dev, dtype=f)  # (4, 3)
        half_size = 0.5 * float(c["keypoint_scale"]) * torch.tensor(
            c["fixed_size"], device=dev, dtype=f
        )  # (3,)
        self._keypoint_offsets = (
            corners.unsqueeze(0) * half_size.unsqueeze(0).unsqueeze(0)
        ).expand(N, -1, -1).contiguous()  # (N, 4, 3)

        # Reward trackers.
        self._lifted_object = torch.zeros(N, dtype=torch.bool, device=dev)
        self._closest_keypoint_max_dist = torch.full((N,), -1.0, device=dev)
        self._closest_fingertip_dist = torch.full((N, NUM_FINGERTIPS), -1.0, device=dev)
        self._successes = torch.zeros(N, dtype=torch.long, device=dev)
        self._near_goal_steps = torch.zeros(N, dtype=torch.long, device=dev)
        self._near_goal = torch.zeros(N, dtype=torch.bool, device=dev)
        self._is_success = torch.zeros(N, dtype=torch.bool, device=dev)
        self._keypoints_max_dist = torch.zeros(N, device=dev)
        self._curr_fingertip_distances = torch.zeros(N, NUM_FINGERTIPS, device=dev)
        self._table_z_per_env = torch.full((N,), float(c["table_reset_z"]), device=dev)
        self._prev_episode_successes = torch.zeros(N, dtype=torch.long, device=dev)
        self.reward_buf = torch.zeros(N, device=dev)

        # Tolerance curriculum.
        self._current_success_tolerance: float = float(c["success_tolerance"])
        self._frame_counter: int = 0
        self._last_curriculum_update: int = 0

        # Domain-randomisation delay queues.
        obs_delay = max(1, int(c["obs_delay_max"]))
        act_delay = max(1, int(c["action_delay_max"]))
        obj_delay = max(1, int(c["object_state_delay_max"]))
        self._obs_queue = torch.zeros(N, obs_delay, self.num_obs, device=dev)
        self._action_queue = torch.zeros(N, act_delay, NUM_JOINTS, device=dev)
        self._object_state_queue = torch.zeros(N, obj_delay, 13, device=dev)

    # ------------------------------------------------------------------
    # RSL-RL interface
    # ------------------------------------------------------------------

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Advance the simulation by one policy step (= *decimation* physics steps).

        Parameters
        ----------
        actions:
            (N, 29) action tensor from the policy.

        Returns
        -------
        obs, rewards, dones, extras
        """
        actions = actions.to(self.device).clamp(
            -float(self.cfg["clip_actions"]), float(self.cfg["clip_actions"])
        )
        self._apply_action(actions)

        for _ in range(self.decimation):
            self.robot.control_dofs_position(self._cur_targets, self._dof_idx)
            self.scene.step()

        self.episode_length_buf += 1

        # Fetch latest simulator state.
        self._update_state()

        # Recompute intermediate geometric values (shared by reward + termination).
        self._compute_intermediate_values()

        # Tolerance curriculum update.
        self._update_tolerance_curriculum()

        # Termination.
        terminated, truncated = self._compute_terminations()
        done = terminated | truncated.bool()

        # Rewards.
        reward = self._compute_rewards()
        self.reward_buf[:] = reward

        # Observations.
        self._build_observations()

        # Fill extras.
        time_out = truncated.float()
        self.extras["time_outs"] = time_out
        self.extras["observations"]["critic"] = self.critic_obs_buf

        # Reset environments that are done.
        reset_ids = done.nonzero(as_tuple=False).flatten()
        if reset_ids.numel() > 0:
            self._prev_episode_successes[reset_ids] = self._successes[reset_ids]
            self.reset_idx(reset_ids)

        return self.obs_buf, reward, done.float(), self.extras

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        self.extras["observations"]["critic"] = self.critic_obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self) -> Optional[torch.Tensor]:
        return None  # privileged obs live in extras["observations"]["critic"]

    def reset(self) -> tuple[torch.Tensor, dict]:
        all_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(all_ids)
        self._build_observations()
        return self.obs_buf, self.extras

    # ------------------------------------------------------------------
    # Reset helpers
    # ------------------------------------------------------------------

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset a subset of environments."""
        if env_ids.numel() == 0:
            return

        self._reset_robot_state(env_ids)
        self._reset_object_pose(env_ids)
        self._reset_goal_pose(env_ids)
        self._reset_goal_trackers(env_ids)

        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = True

    def _reset_robot_state(self, env_ids: torch.Tensor) -> None:
        """Randomise joint positions/velocities and write to sim."""
        c = self.cfg
        n = env_ids.numel()

        lower = self._joint_lower  # (29,)
        upper = self._joint_upper

        # Sample random joint positions within limits.
        rand_pos = lower + (upper - lower) * torch.rand(n, NUM_JOINTS, device=self.device)
        # Interpolate: arm stays close to default (scale=0.1); hand slightly more (0.1).
        scale_arm = float(c["reset_dof_pos_random_interval_arm"])
        scale_hand = float(c["reset_dof_pos_random_interval_fingers"])
        scale = torch.full((NUM_JOINTS,), scale_hand, device=self.device)
        scale[:NUM_ARM_JOINTS] = scale_arm
        default_pos = 0.5 * (lower + upper)  # midpoint as rough "default"
        joint_pos = (1.0 - scale) * default_pos + scale * rand_pos
        joint_pos = joint_pos.clamp(lower, upper)

        vel_range = float(c["reset_dof_vel_random_interval"])
        joint_vel = torch.empty(n, NUM_JOINTS, device=self.device).uniform_(-vel_range, vel_range)

        self.robot.set_dofs_position(
            position=joint_pos,
            dofs_idx_local=self._dof_idx,
            zero_velocity=False,
            envs_idx=env_ids,
        )
        self.robot.set_dofs_velocity(
            velocity=joint_vel,
            dofs_idx_local=self._dof_idx,
            envs_idx=env_ids,
        )
        self._cur_targets[env_ids] = joint_pos
        self._prev_targets[env_ids] = joint_pos

    def _reset_object_pose(self, env_ids: torch.Tensor) -> None:
        """Randomise object start pose and velocity."""
        c = self.cfg
        n = env_ids.numel()
        dev = self.device

        # Table z with small noise.
        dz = torch.empty(n, device=dev).uniform_(-float(c["table_reset_z_range"]),
                                                   float(c["table_reset_z_range"]))
        table_z = float(c["table_reset_z"]) + dz
        self._table_z_per_env[env_ids] = table_z

        obj_z = table_z + float(c["table_object_z_offset"])
        noise_x = torch.empty(n, device=dev).uniform_(-float(c["reset_position_noise_x"]),
                                                        float(c["reset_position_noise_x"]))
        noise_y = torch.empty(n, device=dev).uniform_(-float(c["reset_position_noise_y"]),
                                                        float(c["reset_position_noise_y"]))
        noise_z = torch.empty(n, device=dev).uniform_(-float(c["reset_position_noise_z"]),
                                                        float(c["reset_position_noise_z"]))
        pos = torch.stack([noise_x, noise_y, obj_z + noise_z], dim=-1)  # (n, 3)
        quat = random_orientation(n, dev)                                 # (n, 4) wxyz

        self.object.set_pos(pos, envs_idx=env_ids)
        self.object.set_quat(quat, envs_idx=env_ids)
        self.object.set_vel(torch.zeros(n, 3, device=dev), envs_idx=env_ids)
        self.object.set_ang(torch.zeros(n, 3, device=dev), envs_idx=env_ids)
        self._object_init_z[env_ids] = obj_z

    def _reset_goal_pose(self, env_ids: torch.Tensor) -> None:
        """Sample a new goal pose for the specified envs."""
        c = self.cfg
        n = env_ids.numel()
        dev = self.device
        mode = c["goal_sampling_type"]

        if mode == "absolute":
            mins = tuple(float(v) for v in c["target_volume_mins"])
            maxs = tuple(float(v) for v in c["target_volume_maxs"])
            mins_t = torch.tensor(mins, device=dev)
            maxs_t = torch.tensor(maxs, device=dev)
            pos = mins_t + (maxs_t - mins_t) * torch.rand(n, 3, device=dev)
            quat = random_orientation(n, dev)
        else:
            # Delta goal: perturb previous goal by bounded random walk.
            prev_pos = self._goal_pos[env_ids]
            prev_quat = self._goal_quat[env_ids]
            dist = float(c["delta_goal_distance"])
            deg = float(c["delta_rotation_degrees"])
            mins = tuple(float(v) for v in c["target_volume_mins"])
            maxs = tuple(float(v) for v in c["target_volume_maxs"])
            mins_t = torch.tensor(mins, device=dev)
            maxs_t = torch.tensor(maxs, device=dev)

            pos_noise = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * dist
            pos = torch.clamp(prev_pos + pos_noise, mins_t, maxs_t)

            axis = F.normalize(torch.randn(n, 3, device=dev), dim=-1)
            angle = (torch.rand(n, device=dev) * 2.0 - 1.0) * deg * (math.pi / 180.0)
            dq = quat_from_angle_axis(angle, axis)
            quat = quat_mul(dq, prev_quat)

        self._goal_pos[env_ids] = pos
        self._goal_quat[env_ids] = quat

    def _reset_goal_trackers(self, env_ids: torch.Tensor) -> None:
        """Reset per-env reward / success trackers to initial values."""
        self._lifted_object[env_ids] = False
        self._closest_keypoint_max_dist[env_ids] = -1.0
        self._closest_fingertip_dist[env_ids] = -1.0
        self._successes[env_ids] = 0
        self._near_goal_steps[env_ids] = 0

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def _apply_action(self, actions: torch.Tensor) -> None:
        """Transform raw policy actions into PD position targets."""
        c = self.cfg
        dt = self.dt

        # --- Arm: velocity-delta accumulator ---
        arm_action = actions[:, :NUM_ARM_JOINTS]
        arm_raw = self._prev_targets[:, :NUM_ARM_JOINTS] + float(c["dof_speed_scale"]) * dt * arm_action
        arm_raw = arm_raw.clamp(self._arm_lower, self._arm_upper)
        alpha_arm = float(c["arm_moving_average"])
        arm_smooth = alpha_arm * arm_raw + (1.0 - alpha_arm) * self._prev_targets[:, :NUM_ARM_JOINTS]
        arm_smooth = arm_smooth.clamp(self._arm_lower, self._arm_upper)

        # --- Hand: absolute [-1, 1] to joint range ---
        hand_action = actions[:, NUM_ARM_JOINTS:]
        hand_raw = self._hand_lower + 0.5 * (hand_action + 1.0) * (
            self._hand_upper - self._hand_lower
        )
        alpha_hand = float(c["hand_moving_average"])
        hand_smooth = alpha_hand * hand_raw + (1.0 - alpha_hand) * self._prev_targets[:, NUM_ARM_JOINTS:]
        hand_smooth = hand_smooth.clamp(self._hand_lower, self._hand_upper)

        self._cur_targets[:, :NUM_ARM_JOINTS] = arm_smooth
        self._cur_targets[:, NUM_ARM_JOINTS:] = hand_smooth
        self._prev_targets = self._cur_targets.clone()

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def _update_state(self) -> None:
        """Pull latest simulator state into local buffers."""
        # Robot DOF state.
        self.dof_pos[:] = self.robot.get_dofs_position(self._dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self._dof_idx)

        # Link positions / orientations.
        links_pos = self.robot.get_links_pos()    # (N, n_links, 3)
        links_quat = self.robot.get_links_quat()  # (N, n_links, 4) wxyz
        links_vel = self.robot.get_links_vel()    # (N, n_links, 3) linear
        links_ang = self.robot.get_links_ang()    # (N, n_links, 3) angular

        # Palm state.
        palm_pos_raw = links_pos[:, self._palm_link_idx, :]    # (N, 3)
        palm_quat = links_quat[:, self._palm_link_idx, :]       # (N, 4) wxyz
        palm_lin_vel = links_vel[:, self._palm_link_idx, :]
        palm_ang_vel = links_ang[:, self._palm_link_idx, :]

        # Apply PALM_CENTER_OFFSET in local palm frame.
        offset = torch.tensor(PALM_CENTER_OFFSET, device=self.device, dtype=torch.float32)
        offset_world = quat_apply(palm_quat, offset.unsqueeze(0).expand(self.num_envs, -1))
        self._palm_pos = palm_pos_raw + offset_world           # (N, 3)
        self._palm_quat = palm_quat                             # (N, 4) wxyz
        self._palm_vel = torch.cat([palm_lin_vel, palm_ang_vel], dim=-1)  # (N, 6)

        # Fingertip positions (with small offset toward pad centre).
        ft_offset = torch.tensor(FINGERTIP_OFFSET, device=self.device, dtype=torch.float32)
        ft_idx = self._fingertip_link_idx
        ft_pos_raw = links_pos[:, ft_idx, :]    # (N, 5, 3)
        ft_quat_raw = links_quat[:, ft_idx, :]  # (N, 5, 4)
        ft_offset_exp = ft_offset.unsqueeze(0).unsqueeze(0).expand(self.num_envs, NUM_FINGERTIPS, -1)
        ft_offset_world = quat_apply(
            ft_quat_raw.reshape(-1, 4),
            ft_offset_exp.reshape(-1, 3),
        ).reshape(self.num_envs, NUM_FINGERTIPS, 3)
        self._fingertip_pos = ft_pos_raw + ft_offset_world  # (N, 5, 3)

        # Object state.
        self._object_pos = self.object.get_pos()    # (N, 3)
        self._object_quat = self.object.get_quat()  # (N, 4) wxyz
        self._object_lin_vel = self.object.get_vel()  # (N, 3)
        self._object_ang_vel = self.object.get_ang()  # (N, 3)

    def _compute_intermediate_values(self) -> None:
        """Recompute geometric quantities shared by reward and termination."""
        # Fingertip ↔ object distances.
        self._curr_fingertip_distances = torch.norm(
            self._fingertip_pos - self._object_pos.unsqueeze(1), dim=-1
        )  # (N, 5)

        # Keypoint world positions for object and goal.
        obj_kp = keypoints_world(self._object_pos, self._object_quat, self._keypoint_offsets)
        goal_kp = keypoints_world(self._goal_pos, self._goal_quat, self._keypoint_offsets)
        self._keypoints_max_dist = torch.norm(obj_kp - goal_kp, dim=-1).max(dim=-1).values  # (N,)

        # Initialise closest-so-far sentinels on first use (value = -1).
        sentinel_kp = self._closest_keypoint_max_dist < 0.0
        self._closest_keypoint_max_dist = torch.where(
            sentinel_kp, self._keypoints_max_dist, self._closest_keypoint_max_dist
        )
        sentinel_ft = self._closest_fingertip_dist < 0.0
        self._closest_fingertip_dist = torch.where(
            sentinel_ft, self._curr_fingertip_distances, self._closest_fingertip_dist
        )

        # Success check.
        tol = self._current_success_tolerance * float(self.cfg["keypoint_scale"])
        self._near_goal = self._keypoints_max_dist <= tol
        ng = self._near_goal.long()
        if self.cfg["force_consecutive_near_goal_steps"]:
            self._near_goal_steps = (self._near_goal_steps + ng) * ng
        else:
            self._near_goal_steps = self._near_goal_steps + ng
        self._is_success = self._near_goal_steps >= int(self.cfg["success_steps"])

    # ------------------------------------------------------------------
    # Tolerance curriculum
    # ------------------------------------------------------------------

    def _update_tolerance_curriculum(self) -> None:
        self._frame_counter += 1
        interval = int(self.cfg["tolerance_curriculum_interval"])
        if self._frame_counter - self._last_curriculum_update >= interval:
            mean_succ = self._prev_episode_successes.float().mean().item()
            threshold = 2.0  # require avg 2+ successes/episode before tightening
            if mean_succ >= threshold:
                new_tol = self._current_success_tolerance * float(
                    self.cfg["tolerance_curriculum_increment"]
                )
                lo = float(self.cfg["target_success_tolerance"])
                hi = float(self.cfg["success_tolerance"])
                self._current_success_tolerance = max(min(new_tol, hi), lo)
            self._last_curriculum_update = self._frame_counter

    # ------------------------------------------------------------------
    # Terminations
    # ------------------------------------------------------------------

    def _compute_terminations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(terminated, truncated)`` bool tensors."""
        c = self.cfg
        is_success = self._is_success

        # On goal-reach: increment success counter and re-sample goal.
        self._successes = self._successes + is_success.long()
        goal_reset_ids = is_success.nonzero(as_tuple=False).squeeze(-1)
        if goal_reset_ids.numel() > 0:
            self._reset_goal_pose(goal_reset_ids)
            self._reset_goal_trackers(goal_reset_ids)
            self.episode_length_buf[goal_reset_ids] = 0

        # Episode termination conditions.
        fall = self._object_pos[:, 2] < 0.1
        max_succ = int(c["max_consecutive_successes"])
        if max_succ > 0:
            max_successes_reached = self._successes >= max_succ
        else:
            max_successes_reached = torch.zeros_like(fall)
        hand_far = self._curr_fingertip_distances.max(dim=-1).values > 1.5

        terminated = fall | max_successes_reached | hand_far
        truncated = self.episode_length_buf >= self.max_episode_length
        return terminated, truncated

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _compute_rewards(self) -> torch.Tensor:
        """Sum all reward terms."""
        c = self.cfg
        obj_z = self._object_pos[:, 2]

        # ---- Lifting reward ----
        z_lift = 0.05 + obj_z - self._object_init_z
        lift_progress = z_lift.clamp(0.0, 0.5)
        lifted = (z_lift > float(c["lifting_bonus_threshold"])) | self._lifted_object
        just_lifted = lifted & ~self._lifted_object
        lift_bonus = float(c["lifting_bonus"]) * just_lifted.float()
        lift_rew = lift_progress * (~lifted).float() * float(c["lifting_rew_scale"])
        self._lifted_object = lifted

        # ---- Fingertip approach (before lifted) ----
        ft_deltas = (self._closest_fingertip_dist - self._curr_fingertip_distances).clamp(0.0, 10.0)
        self._closest_fingertip_dist = torch.minimum(
            self._closest_fingertip_dist, self._curr_fingertip_distances
        )
        ft_rew = ft_deltas.sum(dim=-1) * (~lifted).float() * float(c["distance_delta_rew_scale"])

        # ---- Keypoint goal-reach (after lifted) ----
        kp_delta = (self._closest_keypoint_max_dist - self._keypoints_max_dist).clamp(0.0, 100.0)
        self._closest_keypoint_max_dist = torch.minimum(
            self._closest_keypoint_max_dist, self._keypoints_max_dist
        )
        kp_rew = kp_delta * lifted.float() * float(c["keypoint_rew_scale"])

        # ---- Reach-goal bonus ----
        ss = int(c["success_steps"])
        if c["force_consecutive_near_goal_steps"]:
            rg_rew = self._is_success.float() * float(c["reach_goal_bonus"])
        else:
            rg_rew = self._near_goal.float() * (float(c["reach_goal_bonus"]) / ss)

        # ---- Action penalties (joint-velocity L1) ----
        arm_pen = -float(c["kuka_actions_penalty_scale"]) * self.dof_vel[:, :NUM_ARM_JOINTS].abs().sum(dim=-1)
        hand_pen = -float(c["hand_actions_penalty_scale"]) * self.dof_vel[:, NUM_ARM_JOINTS:].abs().sum(dim=-1)

        return lift_rew + lift_bonus + ft_rew + kp_rew + rg_rew + arm_pen + hand_pen

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observations(self) -> None:
        """Populate ``obs_buf`` (actor) and ``critic_obs_buf`` (critic)."""
        c = self.cfg
        N = self.num_envs
        dev = self.device

        # Normalised joint positions ∈ [-1, 1].
        joint_pos_norm = (
            2.0 * (self.dof_pos - self._joint_lower) / self._joint_range - 1.0
        )

        # Joint velocity with optional obs noise.
        joint_vel_obs = self.dof_vel.clone()
        if c["joint_velocity_obs_noise_std"] > 0.0:
            joint_vel_obs += torch.randn_like(joint_vel_obs) * float(c["joint_velocity_obs_noise_std"])

        # Object state — optionally delayed + noisy.
        if c["use_object_state_delay_noise"]:
            obj_pos_obs, obj_rot_obs, obj_vel_obs = self._apply_object_state_dr()
        else:
            obj_pos_obs = self._object_pos
            obj_rot_obs = self._object_quat
            obj_vel_obs = torch.cat([self._object_lin_vel, self._object_ang_vel], dim=-1)

        # Fingertip positions relative to palm centre.
        ft_rel_palm = (self._fingertip_pos - self._palm_pos.unsqueeze(1)).reshape(N, -1)

        # Object keypoints relative to palm.
        obj_kp = keypoints_world(obj_pos_obs, obj_rot_obs, self._keypoint_offsets)
        kp_rel_palm = (obj_kp - self._palm_pos.unsqueeze(1)).reshape(N, -1)

        # Object keypoints relative to goal.
        goal_kp = keypoints_world(self._goal_pos, self._goal_quat, self._keypoint_offsets)
        kp_rel_goal = (obj_kp - goal_kp).reshape(N, -1)

        # Object "scales" observable (all-ones in Stage 1 play; PreciseAssembly
        # overrides this per-env).
        obj_scales = self._object_scale_per_env

        # Build actor obs.
        actor_fields = [
            joint_pos_norm,                               # 29
            joint_vel_obs,                                # 29
            self._prev_targets,                           # 29
            self._palm_pos,                               # 3
            self._palm_quat,                              # 4
            obj_rot_obs,                                  # 4
            ft_rel_palm,                                  # 15
            kp_rel_palm,                                  # 12
            kp_rel_goal,                                  # 12
            obj_scales,                                   # 3
        ]
        actor_obs = torch.cat(actor_fields, dim=-1)

        # Optional obs delay.
        if c["use_obs_delay"] and int(c["obs_delay_max"]) > 0:
            actor_obs = self._apply_obs_delay(actor_obs)

        actor_obs = actor_obs.clamp(-float(c["clamp_abs_observations"]), float(c["clamp_abs_observations"]))
        self.obs_buf[:] = actor_obs

        # Build critic obs (privileged — no delay, no noise).
        critic_fields = [
            joint_pos_norm,                              # 29
            joint_vel_obs,                               # 29
            self._prev_targets,                          # 29
            self._palm_pos,                              # 3
            self._palm_quat,                             # 4
            self._palm_vel,                              # 6
            obj_rot_obs,                                 # 4
            obj_vel_obs,                                 # 6
            ft_rel_palm,                                 # 15
            kp_rel_palm,                                 # 12
            kp_rel_goal,                                 # 12
            obj_scales,                                  # 3
            self._closest_keypoint_max_dist.unsqueeze(-1),   # 1
            self._closest_fingertip_dist,                # 5
            self._lifted_object.float().unsqueeze(-1),   # 1
            (self.episode_length_buf.float() / self.max_episode_length).unsqueeze(-1),  # 1
            self._successes.float().unsqueeze(-1),        # 1
            self.reward_buf.unsqueeze(-1),                # 1
        ]
        critic_obs = torch.cat(critic_fields, dim=-1)
        critic_obs = critic_obs.clamp(-float(c["clamp_abs_observations"]), float(c["clamp_abs_observations"]))
        self.critic_obs_buf[:] = critic_obs

    # ------------------------------------------------------------------
    # Domain-randomisation helpers
    # ------------------------------------------------------------------

    def _apply_obs_delay(self, obs: torch.Tensor) -> torch.Tensor:
        """Push obs into queue and sample a random delay per env."""
        self._obs_queue = torch.roll(self._obs_queue, shifts=1, dims=1)
        self._obs_queue[:, 0, :] = obs
        idx = torch.randint(0, self._obs_queue.shape[1], (self.num_envs,), device=self.device)
        return self._obs_queue[torch.arange(self.num_envs, device=self.device), idx]

    def _apply_object_state_dr(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply delay + pose noise to the observed object state."""
        c = self.cfg
        dev = self.device
        state = torch.cat(
            [self._object_pos, self._object_quat,
             self._object_lin_vel, self._object_ang_vel], dim=-1
        )  # (N, 13)
        self._object_state_queue = torch.roll(self._object_state_queue, shifts=1, dims=1)
        self._object_state_queue[:, 0, :] = state
        idx = torch.randint(0, self._object_state_queue.shape[1], (self.num_envs,), device=dev)
        delayed = self._object_state_queue[torch.arange(self.num_envs, device=dev), idx]

        noisy_pos = delayed[:, 0:3] + torch.randn_like(delayed[:, 0:3]) * float(c["object_state_xyz_noise_std"])
        noisy_rot = perturb_quat(delayed[:, 3:7], float(c["object_state_rotation_noise_degrees"]))
        noisy_vel = delayed[:, 7:13]
        return noisy_pos, noisy_rot, noisy_vel
