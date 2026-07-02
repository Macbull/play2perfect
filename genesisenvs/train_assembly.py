"""Stage-2 Precise-Assembly fine-tuning script for Play2Perfect (Genesis backend).

Loads a Stage-1 play checkpoint as the policy initialisation and fine-tunes
on one of the four assembly tasks using the Genesis simulator.

Supported problems::

    tight_insertion       – L-shaped peg, 0.5 mm tolerance hole
    beam_assembly_step1   – Fabrica beam part 0
    beam_assembly_step2   – Fabrica beam part 2
    screwing              – Furniture-bench table leg

Usage::

    # Fine-tune from a play checkpoint
    python genesisenvs/train_assembly.py \\
        --problem tight_insertion \\
        --checkpoint logs/play/model.pt \\
        --num_envs 1024 \\
        --max_iterations 2000

    # Train from scratch (no play checkpoint)
    python genesisenvs/train_assembly.py --problem tight_insertion

Checkpoints and TensorBoard logs are written to
``logs/assembly_<problem>_<exp_name>/``.

Dependencies
------------
    pip install genesis-world rsl-rl-lib==2.3.3 tensorboard
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
from importlib import metadata
from pathlib import Path

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.3.3":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as exc:
    raise ImportError(
        "Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.3.3'."
    ) from exc

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from genesisenvs.tasks.precise_assembly.precise_assembly_env import (
    PROBLEM_CONFIGS,
    GenesisPreciseAssemblyEnv,
)


# ---------------------------------------------------------------------------
# RSL-RL training config for Stage-2
# ---------------------------------------------------------------------------


def get_train_cfg(exp_name: str, max_iterations: int, checkpoint: str | None) -> dict:
    cfg = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.016,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            # Lower LR for fine-tuning; keeps policy close to the play init.
            "learning_rate": 5e-5,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [1024, 1024, 512, 512],
            "critic_hidden_dims": [1024, 1024, 512, 512],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            # Load the play policy checkpoint if provided.
            "checkpoint": checkpoint if checkpoint else -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": checkpoint is not None,
            "resume_path": checkpoint,
            "run_name": "",
            "logger": "tensorboard",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 16,
        "save_interval": 100,
        "empirical_normalization": False,
        "seed": 42,
    }
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune Play2Perfect Stage-2 (assembly) policy using Genesis."
    )
    parser.add_argument(
        "--problem",
        required=True,
        choices=list(PROBLEM_CONFIGS.keys()),
        help="Assembly task to train on.",
    )
    parser.add_argument(
        "-e", "--exp_name",
        default="",
        help="Optional experiment suffix appended to 'assembly_<problem>_'.",
    )
    parser.add_argument(
        "-B", "--num_envs",
        type=int,
        default=1024,
        help="Number of parallel environments.",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=2000,
        help="PPO update iterations.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to Stage-1 play policy checkpoint (.pt file). "
             "If omitted the policy is initialised from scratch.",
    )
    parser.add_argument(
        "--goal_mode",
        default="preInsertAndFinal",
        choices=["preInsertAndFinal", "finalGoalOnly"],
        help="Goal sampling mode for the assembly task.",
    )
    parser.add_argument(
        "--hole_yaw_range_deg",
        type=float,
        default=0.0,
        help="Per-episode yaw randomisation of the fixture (degrees, symmetric).",
    )
    parser.add_argument("--viewer", action="store_true", help="Show Genesis interactive viewer.")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    suffix = f"_{args.exp_name}" if args.exp_name else ""
    run_name = f"assembly_{args.problem}{suffix}"
    log_dir = os.path.join("logs", run_name)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    train_cfg = get_train_cfg(run_name, args.max_iterations, args.checkpoint)

    with open(os.path.join(log_dir, "cfgs.pkl"), "wb") as f:
        pickle.dump({"problem": args.problem, "train_cfg": train_cfg}, f)

    env = GenesisPreciseAssemblyEnv(
        num_envs=args.num_envs,
        problem=args.problem,
        show_viewer=args.viewer,
        goal_mode=args.goal_mode,
        hole_yaw_range_deg=args.hole_yaw_range_deg,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(
        num_learning_iterations=args.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    main()
