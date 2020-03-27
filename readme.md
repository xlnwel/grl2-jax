**Status:** Active (under active development, breaking changes may occur)

This repository is implemented with *Tensorflow2.0*, and *ray0.8.0.dev6* for distributed learning. Algorithms are tested on *BipedalWalker-v2* and *Atari*.

## Current Implemented Algorithm

- [x] PPO (Proximal Policy Optimization) with LSTM and masking
- [x] PPO2 (different from PPO, PPO2 need explicitly specify lstm states)
- [x] DNC (Differentiable Neural Computer)
- [x] PER (Prioritized Experience Replay)
- [x] MS (Multi Step)
- [x] SAC(Soft Actor-Critic) with adaptive temperature
- [x] Apex-SAC (ApeX with SAC)
- [x] SEED-SAC (SEED with SAC)

**Note**: Here, we only implement a prototype for SEED. To gain the full efficency, one should cooperate it with a larger network(e.g., CNNs). Also, consider separating the action loop from the learner because of the Python GIL.

Also, when impelmenting PER, I intend to omit importance sampling ratio when calculating losses since I found this correction confusing, and in practice, I don't see the real performancre gain for doing so.

## Main Reference Papers

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. “Proximal Policy Optimization Algorithms” 

Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Volodymyr Mnih, Tom Ward, Boron Yotam, et al. 2018. “IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures.” 35th International Conference on Machine Learning, ICML 2018

Alex Graves, Greg Wayne, Malcolm Reynolds, Tim Harley, Ivo Danihelka, Agnieszka Grabska-Barwińska, Sergio Gómez Colmenarejo, et al. 2016. “Hybrid Computing Using a Neural Network with Dynamic External Memory.” Nature 538 (7626). 

Tom Schaul, John Quan, Ioannis Antonoglou, David Silver, and Google Deepmind. 2016. “Prioritized Experience Replay.” ICLR

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. 2018. “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor” 35th International Conference on Machine Learning, ICML 2018

Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, et al. 2018. “Soft Actor-Critic Algorithms and Applications”

Dan Horgan, John Quan, David Budden, Gabriel Barth-Maron, Matteo Hessel, Hado van Hasselt, and David Silver. 2018. “Distributed Prioritized Experience Replay”

Lasse Espeholt, Raphaël Marinier, Piotr Stanczyk, Ke Wang, and Marcin Michalski. 2019. “SEED RL: Scalable and Efficient Deep-RL with Accelerated Central Inference”

Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas,  David Ha, Honglak Lee, James Davidson. Learning Latent Dynamics for Planning from Pixels. In ICML 2019

Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi. Dream to Control: Learning Behaviors by Latent Imagination. In ICLR 2020
