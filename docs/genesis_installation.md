# Genesis Simulator Installation Guide

This document describes how to set up and run the **Genesis simulator backend**
for Play2Perfect.  Use this path if you do not have an Isaac Sim licence or
want a lighter, BSD-3-licensed simulation stack.

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Ubuntu 20.04 / 22.04 (Linux only; Genesis uses Taichi) |
| Python | 3.10 or 3.11 (3.12 is not yet supported by Genesis) |
| GPU | CUDA-capable NVIDIA GPU (≥ 8 GB VRAM recommended for 1 k+ envs) |
| CUDA | 12.x (matching your PyTorch wheel) |
| Disk | ~6 GB for Genesis + dependencies |

---

## 1 – Create a virtual environment

```bash
python3.10 -m venv .venv_genesis
source .venv_genesis/bin/activate
pip install --upgrade pip wheel setuptools
```

---

## 2 – Install PyTorch (CUDA 12)

Pick the wheel matching your CUDA version from <https://pytorch.org/get-started/locally/>.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 3 – Install Genesis

Genesis is installed directly from its GitHub repository:

```bash
pip install git+https://github.com/Genesis-Embodied-AI/Genesis.git
```

This will also pull in Taichi (the CUDA kernel backend) and several geometry
processing libraries (`trimesh`, `libigl`, `tetgen`, …).  Expect the install
to take 5–10 minutes on a fresh environment.

---

## 4 – Install Play2Perfect Genesis dependencies

```bash
cd /path/to/play2perfect
pip install -r requirements_genesis.txt
```

---

## 5 – Quick smoke test

```python
python - <<'EOF'
import genesis as gs
gs.init()
scene = gs.Scene(show_viewer=False)
scene.build(n_envs=4)
scene.step()
print("Genesis smoke test passed.")
EOF
```

---

## 6 – Train Stage 1 (play policy)

```bash
# 4096 envs, headless (default)
python genesisenvs/train_play.py --num_envs 4096 --max_iterations 1500

# With viewer (single GPU, reduces to fewer envs for VRAM)
python genesisenvs/train_play.py --num_envs 512 --viewer
```

TensorBoard logs and checkpoints are written to `logs/play/`.

```bash
tensorboard --logdir logs/play
```

---

## 7 – Train Stage 2 (assembly fine-tune)

```bash
python genesisenvs/train_assembly.py \
    --problem tight_insertion \
    --checkpoint logs/play/model.pt \
    --num_envs 1024 \
    --max_iterations 2000
```

Available problems: `tight_insertion`, `beam_assembly_step1`,
`beam_assembly_step2`, `screwing`.

---

## Differences from the Isaac Sim backend

| Feature | Isaac Sim | Genesis |
|---------|-----------|---------|
| Simulator | NVIDIA Omniverse / Isaac Lab | Genesis (open-source) |
| RL library | rl_games (vendored) | rsl-rl-lib 2.3.3 |
| Config system | Hydra + YAML | Python dicts (inline defaults) |
| Physics solver | PhysX 5 TGS | Genesis rigid solver |
| Licence | Isaac Sim licence required | BSD-3 (fully open-source) |
| Parallel envs | Fabric-backed PhysX | Taichi-backed CUDA kernels |

The Genesis backend re-implements all reward, termination, and observation
computations in pure PyTorch so that trained weights are directly comparable
to the Isaac Lab baseline (same observation and action space dimensions).

---

## Known limitations

* **No multi-USD per-env object diversity** – Stage 1 uses one object URDF
  across all envs per training run.  Run separate training sessions with
  different `--object_urdf` flags to match the full 600-asset diversity of
  the Isaac Lab baseline.
* **External force/torque domain randomisation** – The Genesis backend
  currently omits random impulse injection on the object (present in the
  Isaac Lab DR config).  This can be added via `entity.set_external_force()`
  once Genesis exposes per-link force APIs.
* **No LSTM actor** – The default policy is an MLP.  Swap
  `"class_name": "ActorCriticRecurrent"` and add `rnn_*` keys to the train
  config to enable an LSTM actor.
