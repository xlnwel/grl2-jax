import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

from utility.utils import set_global_seed
from core.tf_config import configure_gpu, configure_precision
from env.gym_env import create_gym_env
from replay.func import create_replay
from replay.data_pipline import DataFormat, Dataset
from algo.run import run, random_sampling
from algo.dreamer.agent import Agent
from algo.dreamer.nn import create_model

LOG_INTERVAL = 4000

def train(agent, env, replay):
    def collect_and_learn(step, **kwargs):
        replay.add(**kwargs)
        if step % 50 == 0:
            for _ in range(50):
                agent.learn_log()

    eval_env = create_gym_env(dict(
        name=env.name, 
        video_path='video',
        log_video=False,
        n_workers=1,
        n_envs=10,
        effective_envvec=True,
        seed=0,
    ))
    start_step = agent.global_steps.numpy() + 1

    print('Training started...')
    step = start_step
    log_step = LOG_INTERVAL
    while step < int(agent.MAX_STEPS):
        agent.set_summary_step(step)
        score, epslen = run(env, agent.actor, fn=collect_and_learn, 
            timer=agent.TIMER, step=step)
        agent.store(score=env.get_score(), epslen=env.get_epslen())
        step += epslen
        
        if step > log_step:
            log_step += LOG_INTERVAL
            agent.save(steps=step)

            
            eval_score, eval_epslen = run(eval_env, agent.actor, 
                evaluation=True, timer=agent.TIMER, name='eval')
            
            agent.store(eval_score=eval_score, eval_epslen=eval_epslen)
            agent.store(**agent.get_value('score', mean=True, std=True, min=True, max=True))
            agent.store(**agent.get_value('epslen', mean=True, std=True, min=True, max=True))
            agent.store(**agent.get_value('eval_score', mean=True, std=True, min=True, max=True))
            agent.store(**agent.get_value('eval_epslen', mean=True, std=True, min=True, max=True))

            agent.log(step)

def main(env_config, model_config, agent_config, replay_config, restore=False, render=False):
    set_global_seed(seed=env_config['seed'], tf=tf)
    # tf.debugging.set_log_device_placement(True)
    configure_gpu()
    configure_precision(agent_config['precision'])

    env = create_gym_env(env_config)

    replay = create_replay(replay_config)
    data_format = dict(
        obs=DataFormat((None, *env.obs_shape), env.obs_dtype),
        action=DataFormat((None, *env.action_shape), env.action_dtype),
        reward=DataFormat((None, ), tf.float32), 
        next_state=DataFormat((None, *env.obs_shape), env.obs_dtype),
        done=DataFormat((None, ), tf.float32),
    )
    if replay_config.get('n_steps', 1) > 1:
        data_format['steps'] = DataFormat((None, ), tf.float32)

    dataset = Dataset(replay, data_format)

    models = create_model(
        model_config, 
        obs_shape=env.obs_shape,
        action_dim=env.action_dim,
        is_action_discrete=env.is_action_discrete
    )

    agent = Agent(name='dreamer',
                config=agent_config,
                models=models, 
                dataset=dataset,
                env=env)

    agent.save_config(dict(
        env=env_config,
        model=model_config,
        agent=agent_config,
        replay=replay_config
    ))

    if restore:
        agent.restore()
        collect_fn = lambda **kwargs: replay.add(**kwargs)      
        while not replay.good_to_learn():
            run(env, agent.actor, collect_fn)
    else:
        random_sampling(env, replay)

    train(agent, env, replay)
