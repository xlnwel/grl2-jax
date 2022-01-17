from env import wrappers


def process_single_agent_env(env, config):
    if config.get('reward_scale') \
            or config.get('reward_min') \
            or config.get('reward_max'):
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


def _change_env_name(config):
    config['env_name'] = config['env_name'].split('-', 1)[-1]
    return config


def make_builtin_gym(config):
    import gym
    from env.dummy import DummyEnv
    config = _change_env_name(config)
    env = gym.make(config['env_name']).env
    env = DummyEnv(env)    # useful for hidding unexpected frame_skip
    config.setdefault('max_episode_steps', 
        env.spec.max_episode_steps)

    env = process_single_agent_env(env, config)
    
    return env


def make_atari(config):
    from env.atari import Atari
    assert 'atari' in config['env_name'], config['env_name']
    config = _change_env_name(config)
    env = Atari(**config)
    config.setdefault('max_episode_steps', 108000)    # 30min
    env = process_single_agent_env(env, config)
    
    return env


def make_procgen(config):
    from env.procgen import Procgen
    assert 'procgen' in config['env_name'], config['env_name']
    gray_scale = config.setdefault('gray_scale', False)
    frame_skip = config.setdefault('frame_skip', 1)
    config = _change_env_name(config)
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
    assert 'dmc' in config['env_name']
    config = _change_env_name(config)
    task = config['env_name']
    env = DeepMindControl(
        task, 
        size=config.setdefault('size', (84, 84)), 
        frame_skip=config.setdefault('frame_skip', 1))
    config.setdefault('max_episode_steps', 1000)
    env = process_single_agent_env(env, config)

    return env


def make_mpe(config):
    from env.mpe_env.MPE_env import MPEEnv
    assert 'mpe' in config['env_name'], config['env_name']
    config = _change_env_name(config)
    env = MPEEnv(config)
    env = wrappers.DataProcess(env)
    env = wrappers.MASimEnvStats(env)

    return env


def make_card(config):
    config = _change_env_name(config)
    env_name = config['env_name']
    if env_name == 'gd':
        from env.guandan.env import Env
        env = Env(**config)
    else:
        raise ValueError(f'No env with env_name({env_name}) is found in card suite')
    env = wrappers.post_wrap(env, config)
    
    return env


def make_smac(config):
    from env.smac import SMAC
    config = config.copy()
    config['env_name'] = config['env_name'].split('_', maxsplit=1)[1]
    env = SMAC(**config)
    env = wrappers.MASimEnvStats(env)
    return env


def make_smac2(config):
    from env.smac2 import SMAC
    config = config.copy()
    config = _change_env_name(config)
    env = SMAC(**config)
    env = wrappers.MASimEnvStats(env)
    return env

def make_overcooked(config):
    assert 'overcooked' in config['env_name'], config['env_name']
    from env.overcooked import Overcooked
    config = config.copy()
    config = _change_env_name(config)
    env = Overcooked(config)
    if config.get('record_state', False):
        env = wrappers.StateRecorder(env, config['rnn_type'], config['state_size'])
    env = wrappers.DataProcess(env)
    env = wrappers.MASimEnvStats(env)
    
    return env
