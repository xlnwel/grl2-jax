from collections import deque
import numpy as np
import ray

from utility.utils import set_global_seed
from utility.tf_utils import configure_gpu
from utility.signal import sigint_shutdown_ray
from utility.timer import Timer
from env.gym_env import create_gym_env
from buffer.replay.proportional_replay import ProportionalPrioritizedReplay
from algo.sac.runner import Runner
from algo.sac.agent import Agent
from algo.sac.eval import evaluate
from algo.sac.data_pipline import Dataset
from algo.sac.nn import create_model


def train(agent, runner, buffer):
    def collect_and_learn(state, action, reward, done):
        buffer.add(state, action, reward, done)
        agent.train_log()

    eval_env = create_gym_env(dict(
        name=runner.env_name, 
        video_path='video',
        log_video=False,
        n_workers=1,
        n_envs=100,
        seed=np.random.randint(100, 10000)
    ))
    log_period = 10
    start_epoch = agent.global_steps.numpy() + 1
    scores = deque(maxlen=100)
    epslens = deque(maxlen=100)
    for epoch in range(start_epoch, agent.n_epochs+1):
        agent.set_summary_step(epoch)
        with Timer('trajectory', log_period):
            score, epslen = runner.sample_trajectory(agent.actor, collect_and_learn)
        scores.append(score)
        epslens.append(epslen)
        
        if epoch % log_period == 0:
            agent.store(
                score=np.mean(scores),
                score_std=np.std(scores),
                epslen=np.mean(epslens),
                epslen_std=np.std(epslens),
            )
            with Timer(f'{agent.model_name} logging'):
                agent.log(epoch, 'Train')
            with Timer(f'{agent.model_name} save'):
                agent.save(steps=epoch)
        if epoch % 100 == 0:
            with Timer(f'{agent.model_name} evaluation'):
                scores, epslens = evaluate(eval_env, agent.actor)
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

    env = create_gym_env(env_config)

    # construct runner
    runner = Runner(env)

    # construct buffer
    buffer = ProportionalPrioritizedReplay(
        buffer_config, env.state_shape, env.action_dim, 
        agent_config['gamma'])
    dataset = Dataset(buffer, env.state_shape, env.action_dim)

    # construct models
    models = create_model(
        model_config, 
        state_shape=env.state_shape, 
        action_dim=env.action_dim, 
        is_action_discrete=env.is_action_discrete)

    # construct agent
    agent = Agent(name='sac', 
                config=agent_config, 
                models=models, 
                dataset=dataset,
                state_shape=env.state_shape,
                state_dtype=env.state_dtype,
                action_dim=env.action_dim,
                action_dtype=env.action_dtype)
    
    agent.save_config(dict(
        env=env_config,
        model=model_config,
        agent=agent_config,
        buffer=buffer_config
    ))

    if restore:
        agent.restore()

    runner.random_sampling(buffer)
    train(agent, runner, buffer)