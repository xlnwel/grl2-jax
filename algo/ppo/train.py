import numpy as np
import ray

from utility.utils import set_global_seed
from core.tf_config import configure_gpu
from utility.signal import sigint_shutdown_ray
from utility.timer import Timer
from env.gym_env import create_gym_env
from buffer.ppo_buffer import PPOBuffer
from algo.ppo.agent import Agent
from algo.ppo.run import run_trajectory
from algo.ppo.eval import evaluate
from algo.ppo.nn import create_model


def train(agent, env, buffer):
    log_period = 10
    start_epoch = agent.global_steps.numpy()+1
    for epoch in range(start_epoch, agent.n_epochs+1):
        agent.set_summary_step(epoch)
        with Timer(f'{agent.model_name} sampling', log_period):
            scores, epslens = run_trajectory(env, agent.ac, buffer)
            score = np.mean(scores)
            agent.store(
                score=score,
                score_std=np.std(scores),
                epslen=np.mean(epslens),
                epslen_std=np.std(epslens)
            )

        with Timer(f'{agent.model_name} training', log_period):
            # TRICK: we only check kl and early terminate the training epoch 
            # when score meets some requirement
            agent.train_epoch(buffer, early_terminate=(agent.max_kl and score > 280), epoch=epoch)

        if epoch % log_period == 0:
            with Timer(f'{agent.model_name} logging'):
                agent.log(epoch, 'Train')
            with Timer(f'{agent.model_name} save'):
                agent.save(steps=epoch)
        # evaluation
        if epoch % 100 == 0:
            with Timer(f'{agent.model_name} evaluation'):
                scores, epslens = evaluate(env, agent.ac)
            stats = dict(
                model_name=f'{agent.model_name}',
                timing='Eval',
                steps=f'{epoch}', 
                score=np.mean(scores),
                score_std=np.std(scores),
                epslen=np.mean(epslens),
                epslen_std=np.std(epslens)
            )
            agent.log_stats(stats)

def main(env_config, model_config, agent_config, buffer_config, restore=False, render=False):
    set_global_seed()
    configure_gpu()

    use_ray = env_config.get('n_workers', 0) > 1
    if use_ray:
        ray.init()
        sigint_shutdown_ray()

    # construct environment
    env = create_gym_env(env_config)
    state_shape = env.state_shape
    action_dim = env.action_dim
    is_action_discrete = env.is_action_discrete
    n_envs = env.n_envs
    
    n_minibatches = agent_config['n_minibatches']
    # construct buffer
    buffer = PPOBuffer(buffer_config, 
                        env.n_envs,
                        env.max_episode_steps,
                        n_minibatches,
                        env.state_shape,
                        env.state_dtype,
                        env.action_shape,
                        env.action_dtype)

    # construct model
    models = create_model(
        model_config, 
        state_shape=state_shape, 
        action_dim=action_dim, 
        is_action_discrete=is_action_discrete,
        n_envs=n_envs
    )
    
    # construct agent for model update
    agent = Agent(name='ppo', 
                config=agent_config, 
                models=models, 
                state_shape=env.state_shape,
                state_dtype=env.state_dtype,
                action_dim=env.action_dim,
                action_dtype=env.action_dtype,
                n_envs=n_envs)

    agent.save_config(dict(
        env=env_config,
        model=model_config,
        agent=agent_config,
        buffer=buffer_config
    ))

    if restore:
        agent.restore()

    train(agent, env, buffer)

    if use_ray:
        ray.shutdown()
