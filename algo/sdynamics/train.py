import numpy as np
import jax

from core.elements.builder import ElementsBuilder
from core.log import do_logging
from core.utils import configure_gpu, set_seed, save_code
from core.typing import dict2AttrDict, ModelPath
from tools.utils import batch_dicts
from tools.timer import Every, Timer
from env.func import create_env
from env.typing import EnvOutput


def run(
    env, 
    n_steps, 
    buffer, 
    env_output, 
):
    with Timer('run'):
        for i in range(n_steps):
            action = env.random_action()
            new_env_output = env.step(action)
            buffer.collect(
                reset=np.concatenate(new_env_output.reset, -1),
                env_state=batch_dicts(new_env_output.env_state,
                    func=lambda x: np.concatenate(x, -2)),
                action=action, 
                next_env_state=batch_dicts(new_env_output.env_state,
                    func=lambda x: np.concatenate(x, -2)),
                reward=np.concatenate(new_env_output.reward, -1), 
                discount=np.concatenate(new_env_output.discount, -1)
            )
            env_output = new_env_output

    return env_output


def split_env_output(env_output):
    env_outputs = [
        jax.tree_util.tree_map(lambda x: x[:, i:i+1], env_output) 
        for i in range(2)
    ]
    return env_outputs


def run_model(model, buffer, routine_config):
    sample_keys = buffer.obs_keys + ['state'] \
        if routine_config.restore_state else buffer.obs_keys 
    obs = buffer.sample_from_recency(
        batch_size=routine_config.n_envs, 
        sample_keys=sample_keys, 
    )
    shape = obs.env_state.shape[-1]
    reward = np.zeros(shape)
    discount = np.ones(shape)
    reset = np.zeros(shape)

    env_output = EnvOutput(obs, reward, discount, reset)
    model.model.choose_elite()
    for i in range(routine_config.n_steps):
        action = np.random.randint(0, 5, (routine_config.n_envs, 2))
        env_output.obs['action'] = action
        _, env_stats = model(env_output)
        model.store(**env_stats)


def record_stats(model, step):
    model.store(**{
        'time/fps': model.get_env_step_intervals() / Timer('run').last(), 
        'time/tps': model.get_train_step_intervals() / Timer('train').last(),
    })
    model.record(step=step)
    model.save()


def train(
    config, 
    model, 
    env, 
    buffer, 
):
    routine_config = config.routine.copy()

    step = model.get_env_step()
    to_record = Every(
        routine_config.LOG_PERIOD, 
        start=step, 
        init_next=step != 0, 
        final=routine_config.MAX_STEPS)

    do_logging('Training starts...')
    env_output = env.output()
    steps_per_iter = env.n_envs * routine_config.n_steps
    while step < routine_config.MAX_STEPS:
        # do_logging(f'start a new iteration with step: {step} vs {routine_config.MAX_STEPS}')
        env_output = run(env, routine_config.n_steps, buffer, env_output)

        step += steps_per_iter
        
        if buffer.ready_to_sample():
            # train the model
            with Timer('train'):
                model.train_record()
            
            run_model(
                model, 
                buffer, 
                routine_config, 
            )
        
            if to_record(step):
                record_stats(model, step)
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
    # save_code(ModelPath(config.root_dir, config.model_name))

    train(config, model, env, buffer)

    do_logging('Training completed')
