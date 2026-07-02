"""Play2Perfect – Genesis simulator backends.

Drop-in Genesis-based training environments for both stages of the
Play2Perfect pipeline.  Uses RSL-RL (rsl-rl-lib==2.3.3) as the RL library,
following the same interface as the pilla_rl reference implementation.

Stage 1 – Play pre-training::

    python genesisenvs/train_play.py --num_envs 4096

Stage 2 – Precise-assembly fine-tune::

    python genesisenvs/train_assembly.py \\
        --checkpoint logs/play/model.pt \\
        --problem tight_insertion \\
        --num_envs 1024
"""
