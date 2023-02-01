import functools
import numpy as np
import ray

from core.elements.builder import ElementsBuilder
from core.log import do_logging
from core.mixin.actor import rms2dict
from core.utils import configure_gpu, set_seed, save_code
from core.typing import dict2AttrDict, ModelPath
from tools.display import print_dict_info
from tools.utils import batch_dicts, modify_config
from tools.timer import Every, Timer
from tools.yaml_op import load_config
from tools import pkg
from jax_tools import jax_utils
from env.func import create_env
from env.typing import EnvOutput


def run(
    env, 
    n_steps, 
    model_collect, 
    env_output, 
):
    for _ in range(n_steps):
        action = env.random_action()
        action = action[0]
        new_env_output = env.step(action)
        model_collect(
            env, 0, 
            reset=np.concatenate(new_env_output.reset, -1),
            obs=batch_dicts(new_env_output.obs, 
                func=lambda x: np.concatenate(x, -2)),
            action=action, 
            next_obs=batch_dicts(new_env_output.obs, 
                func=lambda x: np.concatenate(x, -2))
        )
        env_output = new_env_output

    return env_output


def split_env_output(env_output):
    env_outputs = [
        jax_utils.tree_map(lambda x: x[:, i:i+1], env_output) 
        for i in range(2)
    ]
    return env_outputs


def run_lookahead_agent(model, n_envs, n_steps):
    obs = model.data['obs'].reshape(-1, 2, model.data['obs'].shape[-1])
    obs = obs[np.random.randint(0, obs.shape[0], n_envs)]
    assert len(obs.shape) == 3, obs.shape
    assert obs.shape[:2] == (n_envs, 2), obs.shape
    reward = np.zeros(obs.shape[:-1])
    discount = np.ones(obs.shape[:-1])
    reset = np.zeros(obs.shape[:-1])
    obs = dict2AttrDict({'obs': obs, 'global_state': obs})

    env_output = EnvOutput(obs, reward, discount, reset)
    env_outputs = split_env_output(env_output)
    model.model.choose_elites()
    for i in range(n_steps):
        action = np.random.randint(0, 5, (n_envs, 2))
        env_outputs[0].obs['action'] = action
        _, env_stats = model(env_outputs[0])
        model.store(**env_stats)


def train(
    config, 
    model, 
    env, 
    buffer, 
):
    routine_config = config.routine.copy()
    collect_fn = pkg.import_module(
        'elements.utils', algo=routine_config.algorithm).collect
    collect = functools.partial(collect_fn, buffer)

    step = model.get_env_step()
    to_record = Every(
        routine_config.LOG_PERIOD, 
        start=step, 
        init_next=step != 0, 
        final=routine_config.MAX_STEPS)
    rt = Timer('run')
    tt = Timer('train')

    def record_stats(
        model, step, start_env_step, train_step, start_train_step):
        model.store(**{
            'time/fps': (step-start_env_step)/rt.last(), 
            'time/tps': (train_step-start_train_step)/tt.last(),
        })
        model.record(step=step)
        model.save()

    do_logging('Training starts...')
    env_output = env.output()
    steps_per_iter = env.n_envs * routine_config.n_steps
    while step < routine_config.MAX_STEPS:
        # do_logging(f'start a new iteration with step: {step} vs {routine_config.MAX_STEPS}')
        start_env_step = step
        with rt:
            env_output = run(env, routine_config.n_steps, collect, env_output)

        step += steps_per_iter
        
        # train the model
        start_train_step = model.get_train_step()
        model.train_record()
        train_step = model.get_train_step()
        
        # train lookahead agents
        if routine_config.run_with_future_opponents:
            for _ in range(routine_config.n_lookahead_steps):
                run_lookahead_agent(
                    model, 
                    routine_config.n_lookahead_envs, 
                    routine_config.n_lookahead_steps
                )
        
        if to_record(step):
            record_stats(model, step, start_env_step, train_step, start_train_step)
        # do_logging(f'finish the iteration with step: {step}')


def main(configs, train=train):
    config = configs[0]
    seed = config.get('seed')
    do_logging(f'seed={seed}', level='print')
    set_seed(seed)

    configure_gpu()
    def build_envs():
        env = create_env(config.env, force_envvec=True)

        return env
    
    env = build_envs()

    # load agents
    env_stats = env.stats()
    builder = ElementsBuilder(config, env_stats, to_save_code=False)
    elements = builder.build_agent_from_scratch()
    model = elements.agent
    buffer = elements.buffer
    save_code(ModelPath(config.root_dir, config.model_name))

    train(config, model, env, buffer)

    do_logging('Training completed')
