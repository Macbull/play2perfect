"""Stage-1 Play pre-training script for Play2Perfect (Genesis backend).

This script trains the play policy using the Genesis simulator.  It mirrors
the interface of pilla_rl's ``go2_train.py`` and uses RSL-RL as the RL library.

Usage::

    # Headless training (default 4096 envs, 1500 iterations)
    python genesisenvs/train_play.py

    # Custom settings
    python genesisenvs/train_play.py \\
        --exp_name play_v1 \\
        --num_envs 2048 \\
        --max_iterations 3000 \\
        --object_urdf assets/urdf/handle_head_primitives/screwdriver/screwdriver_0.urdf

Checkpoints and TensorBoard logs are written to ``logs/<exp_name>/``.

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

# Validate rsl-rl version before any heavy imports.
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

# Play2Perfect genesis environment.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from genesisenvs.tasks.play.play_env import GenesisPlayEnv


# ---------------------------------------------------------------------------
# RSL-RL training config (mirrors rsl_config_2_3_3.yaml from pilla_rl)
# ---------------------------------------------------------------------------


def get_train_cfg(exp_name: str, max_iterations: int) -> dict:
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.016,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 1e-4,
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
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "logger": "tensorboard",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 16,
        "save_interval": 100,
        "empirical_normalization": False,
        "seed": 42,
    }


# ---------------------------------------------------------------------------
# Environment config overrides
# ---------------------------------------------------------------------------


def get_env_cfg(args: argparse.Namespace) -> dict:
    cfg: dict = {}
    if args.object_urdf:
        cfg["object_urdf"] = args.object_urdf
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Play2Perfect Stage-1 (play) policy using Genesis."
    )
    parser.add_argument("-e", "--exp_name", default="play", help="Experiment name for logging.")
    parser.add_argument("-B", "--num_envs", type=int, default=4096, help="Number of parallel envs.")
    parser.add_argument("--max_iterations", type=int, default=1500, help="PPO update iterations.")
    parser.add_argument(
        "--object_urdf",
        default=None,
        help="Path to the manipulated object URDF (relative to repo root). "
             "Defaults to hammer_0.urdf.",
    )
    parser.add_argument("--viewer", action="store_true", help="Show Genesis interactive viewer.")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = os.path.join("logs", args.exp_name)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env_cfg = get_env_cfg(args)
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # Save configs for reproducibility.
    with open(os.path.join(log_dir, "cfgs.pkl"), "wb") as f:
        pickle.dump({"env_cfg": env_cfg, "train_cfg": train_cfg}, f)

    env = GenesisPlayEnv(
        num_envs=args.num_envs,
        cfg=env_cfg,
        show_viewer=args.viewer,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(
        num_learning_iterations=args.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    main()
