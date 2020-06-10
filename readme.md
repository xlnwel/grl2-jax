This repository is implemented with *Tensorflow2.1* and *ray0.8.4* for distributed learning. Algorithms are tested with the default configurations.

## Current Implemented Algorithm

- [x] Ape-X
- [x] DNC (Differentiable Neural Computer)
- [x] Dreamer
- [x] GAE
- [x] PPO (with FNN and all algorithmic details following OpenAI's baselines)
- [x] PPO2 (with masked LSTM)
- [x] Rainbow
- [x] IQN
- [x] Retrace(𝝀)
- [x] RND
- [x] SAC with adaptive temperature
- [x] TBO (Transformed Bellman Operator)
- [x] SEED-Dreamer

## Get Started

```
python run/train.py algo -e env
```
For available `algo`, please refer to the folder names in `/algo`. To run distributed algorithms, `algo` should be of form `distributed_architecture-algorithm`. For example, if you want to run Ape-X with DQN, replace 'algo' with `apex-dqn`. Currently supported distributed algorithms only include `apex-sac`, `apex-dqn`, `seed-dreamer`.


## Reference Papers

Graves, Alex, Greg Wayne, Malcolm Reynolds, Tim Harley, Ivo Danihelka, Agnieszka Grabska-Barwińska, Sergio Gómez Colmenarejo, et al. 2016. “Hybrid Computing Using a Neural Network with Dynamic External Memory.” Nature 538 (7626): 471–76. https://doi.org/10.1038/nature20101.

Burda, Yuri, Harrison Edwards, Amos Storkey, and Oleg Klimov. 2018. “Exploration by Random Network Distillation,” 1–17. http://arxiv.org/abs/1810.12894.

Horgan, Dan, John Quan, David Budden, Gabriel Barth-Maron, Matteo Hessel, Hado van Hasselt, and David Silver. 2018. “Distributed Prioritized Experience Replay.” In ICLR, 1–19. http://arxiv.org/abs/1803.00933.

Munos, Rémi, Thomas Stepleton, Anna Harutyunyan, and Marc G. Bellemare. 2016. “Safe and Efficient Off-Policy Reinforcement Learning.” Advances in Neural Information Processing Systems, no. Nips: 1054–62.

Engstrom, Logan, Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Firdaus Janoos, Larry Rudolph, and Aleksander Madry. 2019. “Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO.” ICLR

Haarnoja, Tuomas, Aurick Zhou, Pieter Abbeel, and Sergey Levine. 2018. “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.” 35th International Conference on Machine Learning, ICML 2018 5: 2976–89.

Kapturowski, Steven, Georg Ostrovski, John Quan, and Will Dabney. 2019. “Recurrent Experience Replay in Distributed Reinforcement Learning.” In ICLR, 1–19.

Jaderberg, Max, Wojciech M. Czarnecki, Iain Dunning, Luke Marris, Guy Lever, Antonio Garcia Castañeda, Charles Beattie, et al. 2019. “Human-Level Performance in 3D Multiplayer Games with Population-Based Reinforcement Learning.” Science 364 (6443): 859–65. https://doi.org/10.1126/science.aau6249.

Haarnoja, Tuomas, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, et al. 2018. “Soft Actor-Critic Algorithms and Applications.” http://arxiv.org/abs/1812.05905.

Espeholt, Lasse, Hubert Soyer, Remi Munos, Karen Simonyan, Volodymyr Mnih, Tom Ward, Boron Yotam, et al. 2018. “IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures.” 35th International Conference on Machine Learning, ICML 2018 4: 2263–84.

Espeholt, Lasse, Raphaël Marinier, Piotr Stanczyk, Ke Wang, and Marcin Michalski. 2019. “SEED RL: Scalable and Efficient Deep-RL with Accelerated Central Inference,” 1–19. http://arxiv.org/abs/1910.06591.

Pardo, Fabio, Arash Tavakoli, Vitaly Levdik, and Petar Kormushev. 2018. “Time Limits in Reinforcement Learning.” 35th International Conference on Machine Learning, ICML 2018 9: 6443–52.

Machado, Marlos C., Marc G. Bellemare, Erik Talvitie, Joel Veness, Matthew Hausknecht, and Michael Bowling. 2018. “Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents.” IJCAI International Joint Conference on Artificial Intelligence 2018-July (2013): 5573–77.

Hafner, Danijar, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. 2020. “Dream to Control: Learning Behaviors by Latent Imagination.” ICLR, 1–20. http://arxiv.org/abs/1912.01603.

Hafner, Danijar, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. 2019. “Learning Latent Dynamics for Planning from Pixels.” 36th International Conference on Machine Learning, ICML 2019 2019-June: 4528–47.

## Reference Repository

https://github.com/openai/baselines

https://github.com/google/dopamine

https://github.com/deepmind/dnc

https://github.com/danijar/dreamer

https://github.com/google-research/seed_rl

## Acknowledge

I'd like to especially thank @danijar for his great help with Dreamer.
