from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from core.tf_config import *
from utility.run import evaluate
from utility.display import pwc
from utility.graph import save_video
from utility import pkg
from env.func import create_env


def main(env_config, model_config, agent_config, n, record=False, size=(128, 128), video_len=1000, fps=30):
    silence_tf_logs()
    configure_gpu()
    configure_precision(agent_config.get('precision', 32))

    use_ray = env_config.get('n_workers', 0) > 1
    if use_ray:
        import ray
        from utility.ray_setup import sigint_shutdown_ray
        ray.init()
        sigint_shutdown_ray()

    algo_name = agent_config['algorithm']
    env_name = env_config['name']

    try:
        make_env = pkg.import_module('env', algo_name, place=-1).make_env
    except:
        make_env = None
    env_config.pop('reward_clip', False)
    env_config['n_envs'] = 1
    env = create_env(env_config, env_fn=make_env)
    create_model, Agent = pkg.import_agent(config=agent_config)    
    models = create_model(model_config, env)

    agent = Agent( 
        config=agent_config, 
        models=models, 
        dataset=None, 
        env=env)

    if n < env.n_envs:
        n = env.n_envs

    stats = defaultdict(list)
    def step_fn(**kwargs):
        for k, v in kwargs.items():
            stats[k].append(v)
    scores, epslens, video = evaluate(
        env, 
        agent, 
        n, 
        record=record, 
        size=size, 
        video_len=video_len, 
        step_fn=step_fn,
        record_stats=True)
    pwc(f'After running {n} episodes',
        f'Score: {np.mean(scores):.3g}\tEpslen: {np.mean(epslens):.3g}', color='cyan')
    
    seqlen = len(stats['reward'])
    x_0 = np.arange(seqlen)

    sns.set(style="whitegrid", font_scale=1.5)
    sns.set_palette('Set2') # or husl

    reward = stats['reward']
    q = stats['q']
    q_max = stats['q_max']
    x = np.tile(x_0, 3)
    y = np.array(reward + q + q_max)
    tag = ['reward' for _ in reward] + ['q' for _ in q] + ['q_max' for _ in q_max]

    fig = plt.figure(figsize=(10*2, 10))
    ax = fig.add_subplot(1, 2, 1)
    sns.lineplot(x=x, y=y, ax=ax, hue=tag)
    ax.grid(True, alpha=0.8, linestyle=':')
    ax.legend(loc='best').set_draggable(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    action = stats['action']
    action_best_q = stats['action_best_q']
    x = np.tile(x_0, 2)
    y = np.array(action + action_best_q)
    tag = ['action' for _ in action] + ['action_best_q' for _ in action_best_q]

    ax = fig.add_subplot(1, 2, 2)
    sns.scatterplot(x=x, y=y, ax=ax, hue=tag)
    ax.grid(True, alpha=0.8, linestyle=':')
    ax.legend(loc='best').set_draggable(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # fig.show()
    name = f'{algo_name}-{env_name}-{np.sum(reward)}'
    fig.savefig(f'results/{name}.png')

    if record:
        save_video(name, video, fps=fps)
    if use_ray:
        ray.shutdown()