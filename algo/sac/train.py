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
from algo.sac.data_pipline import Dataset
from algo.sac import nn
from algo.sac.agent import Agent
from algo.ppo.eval import evaluate


def train(agent, runner, buffer):
    def collect_and_learn(state, action, reward, done):
        buffer.add(state, action, reward, done)
        agent.train_log()

    eval_env = create_gym_env(dict(
        name=runner.env.name, 
        video_path='video',
        log_video=False,
        n_workers=1,
        n_envs=100,
        seed=0
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

def main(env_config, model_config, agent_config, buffer_config, render=False):
    set_global_seed()
    configure_gpu()

    env = create_gym_env(env_config)
    state_shape = env.state_shape
    action_dim = env.action_dim
    is_action_discrete = env.is_action_discrete

    # construct runner
    runner = Runner(env)

    # construct buffer
    buffer = ProportionalPrioritizedReplay(
        buffer_config, state_shape, action_dim, 
        agent_config['gamma'])
    dataset = Dataset(buffer, state_shape, action_dim)
    
    # construct models
    actor_config = model_config['actor']
    softq_config = model_config['softq']
    temperature_config = model_config['temperature']
    actor = nn.SoftPolicy(actor_config, state_shape, action_dim, is_action_discrete, 'actor')
    softq1 = nn.SoftQ(softq_config, state_shape, action_dim, 'softq1')
    softq2 = nn.SoftQ(softq_config, state_shape, action_dim, 'softq2')
    target_softq1 = nn.SoftQ(softq_config, state_shape, action_dim, 'target_softq1')
    target_softq2 = nn.SoftQ(softq_config, state_shape, action_dim, 'target_softq2')
    temperature = nn.Temperature(temperature_config, state_shape, action_dim, 'temperature')
    models = [actor, softq1, softq2, target_softq1, target_softq2, temperature]

    # construct agent
    agent = Agent(name='sac', 
                config=agent_config, 
                models=models, 
                dataset=dataset,
                state_shape=env.state_shape,
                state_dtype=env.state_dtype,
                action_dim=env.action_dim,
                action_dtype=env.action_dtype,)

    agent.restore()

    runner.random_sampling(buffer)
    train(agent, runner, buffer)