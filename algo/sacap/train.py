from collections import deque
import numpy as np
import tensorflow as tf
import ray

from utility.utils import set_global_seed
from core.tf_config import configure_gpu
from utility.signal import sigint_shutdown_ray
from utility.timer import Timer
from utility.utils import step_str
from env.gym_env import create_gym_env
from replay.func import create_replay
from replay.data_pipline import Dataset
from algo.run import run, random_sampling
from algo.sac.agent import Agent
from algo.sac.nn import create_model


LOG_INTERVAL = 10000

def run_trajectory(env, actor, *, fn=None, evaluation=False, 
                    timer=False, step=0, render=False):
    """ Sample a trajectory

    Args:
        env: an Env instance
        actor: the model responsible for taking actions
        fn: a function that specifies what to do after each env step
        step: environment step
    """
    action_fn = actor.det_action if evaluation else actor.action

    while True:
        state = env.reset()
        for i in range(1, env.max_episode_steps+1):
            if render:
                env.render()
            state_expanded = np.expand_dims(state, 0)
            
            action, n_ar = action_fn(tf.convert_to_tensor(state_expanded, tf.float32)).numpy()[0]
            rewards = 0
            for i in range(1, n_ar+1):
                next_state, reward, done, _ = env.step(action)
                rewards += reward
                if done:
                    n_ar = i
                    break
            if fn:
                fn(state=state, action=action, reward=rewards, 
                    done=done, next_state=next_state, 
                    step=step+i, action_rep=n_ar)
            state = next_state
            if done:
                break
        if env.already_done:
            break
        else:
            print(f'not already done, {env.get_epslen()}')
        
    return env.get_score(), env.get_epslen()

def train(agent, env, replay):
    def collect_and_learn(state, action, reward, done, next_state, action_rep, **kwargs):
        replay.add(state=state, action=action, reward=reward, 
            done=done, next_state=next_state, action_rep=action_rep)
        agent.learn_log()

    eval_env = create_gym_env(dict(
        name=env.name, 
        video_path='video',
        log_video=False,
        n_workers=1,
        n_envs=50,
        seed=np.random.randint(100, 10000)
    ))
    start_step = agent.global_steps.numpy() + 1
    scores = deque(maxlen=100)
    epslens = deque(maxlen=100)
    print('Training started...')
    step = start_step
    log_step = LOG_INTERVAL
    while step < int(agent.max_steps):
        agent.set_summary_step(step)
        with Timer(f'{agent.model_name}: trajectory', agent.LOG_INTERVAL):
            score, epslen = run(env, agent.actor, collect_and_learn, action_rep=True)
        step += epslen
        scores.append(score)
        epslens.append(epslen)
        
        if step > log_step:
            log_step += LOG_INTERVAL
            agent.save(steps=step)

            with Timer(f'{agent.model_name} evaluation'):
                eval_scores, eval_epslens = run(eval_env, agent.actor, evaluation=True, action_rep=True)
            agent.store(
                score=np.mean(eval_scores),
                score_std=np.std(eval_scores),
                score_max=np.max(eval_scores),
                epslen=np.mean(eval_epslens),
                epslen_std=np.std(eval_epslens)
            )
            agent.log(step, timing='Eval')

            stats = dict(
                model_name=f'{agent.model_name}',
                timing='Train',
                steps=step_str(step), 
                score=np.mean(scores),
                score_std=np.std(scores),
                score_max=np.max(scores),
                epslen=np.mean(epslens),
                epslen_std=np.std(epslens),
            )
            agent.log_stats(stats)

def main(env_config, model_config, agent_config, replay_config, restore=False, render=False):
    set_global_seed()
    configure_gpu()

    env = create_gym_env(env_config)

    # construct replay
    replay_keys = ['state', 'action', 'reward', 'done', 'steps', 'action_rep']
    replay = create_replay(replay_config, *replay_keys, state_shape=env.state_shape)
    dataset = Dataset(replay, env.state_shape, env.state_dtype, env.action_shape, env.action_dtype)

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
                env=env)
    
    agent.save_config(dict(
        env=env_config,
        model=model_config,
        agent=agent_config,
        replay=replay_config
    ))

    if restore:
        agent.restore()
        collect_fn = (
            lambda state, action, reward, done, next_state, **kwargs: 
            replay.add(state=state, action=action, 
            reward=reward, done=done, next_state=next_state))        
        while not replay.good_to_learn():
            run(env, agent.actor, collect_fn)
    else:
        random_sampling(env, replay)

    train(agent, env, replay)
