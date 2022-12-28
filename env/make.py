import gym
from core.typing import dict2AttrDict

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
    if isinstance(env.action_space, gym.spaces.Box):
        env = wrappers.ContinuousActionMapper(
            env, 
            bound_method=config.get('bound_method', 'clip'), 
            to_rescale=config.get('to_rescale', True),
            action_low=config.get('action_low', -1), 
            action_high=config.get('action_high', 1)
        )
    if config.get('to_multi_agent', False):
        env = wrappers.Single2MultiAgent(env)
    env = wrappers.post_wrap(env, config)

    return env


def _change_env_name(config):
    config = dict2AttrDict(config, to_copy=True)
    config['env_name'] = config['env_name'].split('-', 1)[-1]
    return config


def make_bypass(config):
    from env.bypass import BypassEnv
    config = _change_env_name(config)
    env = BypassEnv()
    env = process_single_agent_env(env, config)

    return env

def make_gym(config):
    import gym
    from env.dummy import DummyEnv
    config = _change_env_name(config)
    env = gym.make(config['env_name']).env
    env = DummyEnv(env)    # useful for hidding unexpected frame_skip
    config.setdefault('max_episode_steps', env.spec.max_episode_steps)
    env = process_single_agent_env(env, config)

    return env


def make_sagw(config):
    from env.sagw import env_map
    config = _change_env_name(config)
    env = env_map[config['env_name']](**config)
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


def make_spiel(config):
    config = _change_env_name(config)
    from env.openspiel import OpenSpiel
    env = OpenSpiel(**config)
    env = wrappers.TurnBasedProcess(env)
    # env = wrappers.SqueezeObs(env, config['squeeze_keys'])
    env = wrappers.MATurnBasedEnvStats(env)

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
    config = _change_env_name(config)
    env = SMAC(**config)
    env = wrappers.MASimEnvStats(env)

    return env


def make_smac2(config):
    from env.smac2 import SMAC
    config = _change_env_name(config)
    env = SMAC(**config)
    env = wrappers.MASimEnvStats(env)

    return env

def make_overcooked(config):
    assert 'overcooked' in config['env_name'], config['env_name']
    from env.overcooked import Overcooked
    config = _change_env_name(config)
    env = Overcooked(config)
    if config.get('record_state', False):
        env = wrappers.StateRecorder(env, config['rnn_type'], config['state_size'])
    env = wrappers.DataProcess(env)
    env = wrappers.MASimEnvStats(env)
    
    return env

def make_matrix(config):
    assert 'matrix' in config['env_name'], config['env_name']
    from env.matrix import env_map
    config = _change_env_name(config)
    env = env_map[config['env_name']](**config)
    env = wrappers.MultiAgentUnitsDivision(env, config['uid2aid'])
    env = wrappers.DataProcess(env)
    env = wrappers.MASimEnvStats(env)

    return env

def make_magw(config):
    assert 'magw' in config['env_name'], config['env_name']
    from env.magw import env_map
    config = _change_env_name(config)
    env = env_map[config['env_name']](**config)
    env = wrappers.MultiAgentUnitsDivision(env, config['uid2aid'])
    env = wrappers.PopulationSelection(env, config.pop('population_size', 1))
    env = wrappers.DataProcess(env)
    env = wrappers.MASimEnvStats(env, timeout_done=config.get('timeout_done', True))

    return env

def make_smarts(config):
    assert 'smarts' in config['env_name'], config['env_name']
    from env.hn_smarts import make
    config = _change_env_name(config)
    env = make(config)
    env = wrappers.DataProcess(env)
    env = wrappers.MASimEnvStats(env)

    return env

def make_grf(config):
    assert 'grf' in config['env_name'], config['env_name']
    from env.grf import GRF
    config = _change_env_name(config)
    env = GRF(**config)
    env = wrappers.DataProcess(env)
    env = wrappers.MASimEnvStats(env)

    return env

def make_unity(config):
    from env.unity import Unity
    config = _change_env_name(config)
    env = Unity(config)
    env = wrappers.ContinuousActionMapper(
        env, 
        bound_method=config.get('bound_method', 'clip'), 
        to_rescale=config.get('to_rescale', True),
        action_low=config.get('action_low', -1), 
        action_high=config.get('action_high', 1)
    )
    env = wrappers.UnityEnvStats(env)

    return env

if __name__ == '__main__':
    from tools import yaml_op
    config = yaml_op.load_config('algo/zero/configs/mpe')
    env = make_mpe(config.env)
    print(env.action_shape)
    for _ in range(1):
        a = env.random_action()
        o, r, d, re = env.step(a)
        print(o)
        if re:
            print('discount at reset', d)
            print('epslen', env.epslen())