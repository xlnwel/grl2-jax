**Status:** Active (under active development, breaking changes may occur)

This repository is implemented using *Tensorflow2.0* and *ray0.8.0.dev6* for distributed learning. Algorithms are tested on *BipedalWalker-v2*.

## Current Implemented Algorithm

- [x] PPO (Proximal Policy Optimization) with LSTM and masking
- [x] PPO2 (different from PPO, PPO2 need explicitly specify lstm states)
- [x] DNC (Differentiable Neural Computer)
- [x] PER (Prioritized Experience Replay)
- [x] MS (Multi Step)
- [x] SAC(Soft Actor-Critic) with adaptive temperature
- [x] Apex-SAC (Apex with SAC)
- [x] SEED-SAC (SEED with SAC)
