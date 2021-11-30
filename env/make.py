from env import wrappers
from env.utils import process_single_agent_env


def make_builtin_gym(config):
    import gym
    from env.dummy import DummyEnv
    config['name'] = config['name'].split('_')[-1]
    env = gym.make(config['name']).env
    env = DummyEnv(env)    # useful for hidding unexpected frame_skip
    config.setdefault('max_episode_steps', 
        env.spec.max_episode_steps)

    env = process_single_agent_env(env, config)
    
    return env


def make_atari(config):
    from env.atari import Atari
    
    assert 'atari' in config['name'], config['name']
    env = Atari(**config)
    config.setdefault('max_episode_steps', 108000)    # 30min
    env = process_single_agent_env(env, config)
    
    return env


def make_procgen(config):
    from env.procgen import Procgen
    assert 'procgen' in config['name'], config['name']
    gray_scale = config.setdefault('gray_scale', False)
    frame_skip = config.setdefault('frame_skip', 1)
    env = Procgen(config)
    if gray_scale:
        env = wrappers.GrayScale(env)
    if frame_skip > 1:
        if gray_scale:
            env = wrappers.MaxAndSkipEnv(env, frame_skip=frame_skip)
        else:
            env = wrappers.FrameSkip(env, frame_skip=frame_skip)
    config.setdefault('max_episode_steps', env.spec.max_episode_steps)
    if config['max_episode_steps'] is None:
        config['max_episode_steps'] = int(1e9)
    env = process_single_agent_env(env, config)
    
    return env


def make_dmc(config):
    from env.dmc import DeepMindControl
    assert 'dmc' in config['name']
    task = config['name'].split('_', 1)[-1]
    env = DeepMindControl(
        task, 
        size=config.setdefault('size', (84, 84)), 
        frame_skip=config.setdefault('frame_skip', 1))
    config.setdefault('max_episode_steps', 1000)
    env = process_single_agent_env(env, config)

    return env


def make_mpe(config):
    from env.mpe_env.MPE_env import MPEEnv
    assert 'mpe' in config['name'], config['name']
    env = MPEEnv(config)
    env = wrappers.DataProcess(env)
    env = wrappers.MASimEnvStats(env)

    return env


def make_card(config):
    from env.guandan.env import Env as Env1
    from env.guandan2.env import Env as Env2
    name = config['name'].split('_', 1)[1]
    if name == 'gd':
        env = Env1(**config)
    elif name == 'gd2':
        env = Env2(**config)
    else:
        raise ValueError(f'No env with name({name}) is found in card suite')
    env = wrappers.post_wrap(env, config)
    
    return env


def make_gd2(config):
    from env.guandan2.env import Env
    env = Env(config['eid'])
    env = wrappers.post_wrap(env, config)
    
    return env


def make_smac(config):
    from env.smac import SMAC
    config = config.copy()
    config['name'] = config['name'].split('_', maxsplit=1)[1]
    env = SMAC(**config)
    env = wrappers.MASimEnvStats(env)
    return env


def make_smac2(config):
    from env.smac2 import SMAC
    config = config.copy()
    config['name'] = config['name'].split('_', maxsplit=1)[1]
    env = SMAC(**config)
    env = wrappers.MASimEnvStats(env)
    return env
