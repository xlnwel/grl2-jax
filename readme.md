## Instructions

If you want to know how an algorithm works, simply study each folder in [algo](https://github.com/xlnwel/g2rl/tree/master/algo).

If you want to run some algorithm, refer to [Get Start](#start).

## <a name="start"></a>Get Started

### Training

```shell
python run/train.py -a sync_sim-hm -e unity-combat2d
```

where `sync_sim` specifies the distributed architecture(dir: distributed), `hm` specifies the algorithm(dir: algo), `unity` denotes the environment suite, and `combat2d` is the environment name

By default, all the checkpoints and loggings are saved in `./logs/{env}/{algo}/{model_name}/`.

You can also make some simple changes to `*.yaml` from the command line

```shell
# change learning rate to 0.0001, `lr` must appear in `*.yaml`
python run/train.py -a sync_sim-hm -e unity-combat2d -kw lr=0.0001
```

This change will automatically be embodied in Tensorboard, making it a recommanded way to do some simple hyperparameter tuning. Alternatively, you can modify configurations in `*.yaml` and specify `model_name` manually using command argument `-n your_model_name`.

The following code demonstrates an example of running multi-agent algorithms with multiple configs

```shell
# make sure `unity.yaml` and `unity2.yaml` exist in `configs/` directory
python run/train.py -a sync_sim-hm -e unity-combat2d -c unity unity2
```
