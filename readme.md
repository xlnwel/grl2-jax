## Overview

This repository is a multi-agent reinforcement learning repository for human-machine coorperation games. 

Currently, we only implement PPO and synchronous training for simultaneous-step environments. Basic algorithms are implemented in folder "algo", and the distributed architectures are implemented in folder "distributed". 


## <a name="start"></a>Get Started

### Training

#### A Robust Way for Training with Error-Prone Simulators

All the following `python run/train.py` can be replaced by `python main.py`, which automatically detects unexpected halts caused by simulator errors and restarts the whole system accordingly. 

For stable simulators, `python run/train.py` is still the recommanded way to go.

#### Basics

```shell
python run/train.py -a sync-hm -e unity-combat2d
```

where `sync` specifies the distributed architecture(dir: distributed), `hm` specifies the algorithm(dir: algo), `unity` denotes the environment suite, and `combat2d` is the environment name

By default, all the checkpoints and loggings are saved in `./logs/{env}/{algo}/{model_name}/`.

#### Several Useful Commandline Arguments

You can also make some simple changes to `*.yaml` from the command line

```shell
# change learning rate to 0.0001, `lr` must appear in `*.yaml`
python run/train.py -a sync-hm -e unity-combat2d -kw lr=0.0001
```

This change will automatically be embodied in Tensorboard, making it a recommanded way to do some simple hyperparameter tuning. Alternatively, you can modify configurations in `*.yaml` and specify `model_name` manually using command argument `-n your_model_name`.

#### Training Multiple Agents with Different Configurations

In some multi-agent settings, we may prefer using different configurations for different agents. The following code demonstrates an example of running multi-agent algorithms with multiple configurations, one for each agent.

```shell
# make sure `unity.yaml` and `unity2.yaml` exist in `configs/` directory
# the first agent is initialized with the configuration specified by `unity.yaml`, 
# while the second agent is initialized with the configuration specified by `unity2.yaml`
python run/train.py -a sync-hm -e unity-combat2d -c unity unity2
```
