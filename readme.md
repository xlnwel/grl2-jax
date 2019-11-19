**Status:** Active (under active development, breaking changes may occur)

This repository is implemented based on *Tensorflow2.0*. Algorithms are tested on *BipedalWalker-v2*.

## Current Implemented Algorithm

- [x] PPO(Proximal Policy Optimization) with LSTM and masking
- [x] DNC(Differentiable Neural Computer)

**Note**: DNC takes significantly more time to run and it does not contribute to solving *BipedalWalker-v2* -- even without DNC, PPO with LSTM can solve it, which is reasonable since there is no relational relationship involved here. 
