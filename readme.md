## Overview

A multi-agent reinforcement learning library. 

## 概述

这是一个模块化的分布式多智能体强化学习的框架. 它主要由三个模块构成: i) 单/多智能体算法, ii) 分布式训练框架, iii) 博弈. 本文先介绍框架的使用指南, 然后再依次阐述这三个模块设计.

## 优势

- 容易上手, 不需要编程基础也能在半小时内轻松学会多机调参实验. 
- 模块化设计, 方便扩展, 新算法和环境的引入只需要遵循预先定制的接口, 即可即插即用.
- 现有的基础算法在多个benchmark上取得了SOTA的水平, 包括SMAC, GRF等经典的多智能体测试环境. 
- 分布式训练框架, 支持自博弈以及不对称的多种群博弈, 评估.

## 使用指南

## 单/多智能体算法

单/多智能体算法的入口在algo/train.py, 算法由Agent定义, 大部分的交互模块都定义在Runner这个类里.

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

#### Evaluation

```shell
python run/eval.py magw-logs/n_envs=64-n_steps=20-n_epochs=1/seed=4/ -n 1 -ne 1 -nr 1 -r -i eval -s 256 256 --fps 1
```

The above code presents a way for evaluating a trained model, where

- `magw-logs/n_envs=64-n_steps=20-n_epochs=1/seed=4/` is the model path
- `-n` specifies the number of eposodes to run
- `-ne` specifies the number of environments running in parallel
- `-nr` specifies the number of ray actors are devoted for runniing
- `-r` visualizes the video and save it as a `*.gif` file
- `-i` specifies the video name
- `-s` specifies the screen size of the video
- `--fps` specifies the fps of the saved `*.gif` file

#### Training Multiple Agents with Different Configurations

In some multi-agent settings, we may prefer using different configurations for different agents. The following code demonstrates an example of running multi-agent algorithms with multiple configurations, one for each agent.

```shell
# make sure `unity.yaml` and `unity2.yaml` exist in `configs/` directory
# the first agent is initialized with the configuration specified by `unity.yaml`, 
# while the second agent is initialized with the configuration specified by `unity2.yaml`
python run/train.py -a sync-hm -e unity-combat2d -c unity unity2
```
