from env import wrappers
from env.mpe import make_mpe
from env.smac import make_smac
from env.smac2 import make_smac2
from env2.smac3 import make_smac3
from env2.smac4 import make_smac4
from env.atari import make_atari
from env.procgen import make_procgen
from env.dmc import make_dmc
from env.builtin import make_built_in_gym


def process_single_agent_env(env, config):
    if config.get('reward_scale') or config.get('reward_clip'):
        env = wrappers.RewardHack(env, **config)
    frame_stack = config.setdefault('frame_stack', 1)
    if frame_stack > 1:
        np_obs = config.setdefault('np_obs', False)
        env = wrappers.FrameStack(env, frame_stack, np_obs)
    frame_diff = config.setdefault('frame_diff', False)
    assert not (frame_diff and frame_stack > 1), f"Don't support using FrameStack and FrameDiff at the same time"
    if frame_diff:
        gray_scale_residual = config.setdefault('gray_scale_residual', False)
        distance = config.setdefault('distance', 1)
        env = wrappers.FrameDiff(env, gray_scale_residual, distance)
    env = wrappers.post_wrap(env, config)

    return env


def make_env(config):
    config = config.copy()
    env_name = config['name'].lower()
    env_func = dict(
        mpe=make_mpe,
        smac=make_smac,
        smac2=make_smac2,
        smac3=make_smac3,   # still under test
        smac4=make_smac4,   # still under test
        atari=make_atari,
        procgen=make_procgen,
        dmc=make_dmc,
    ).get(env_name.split('_', 1)[0], make_built_in_gym)
    env = env_func(config)
    
    return env
