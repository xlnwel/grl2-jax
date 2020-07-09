import time
import numpy as np
import ray 

from core.tf_config import configure_gpu, configure_precision, silence_tf_logs
from utility.utils import Every, TempStore
from utility.ray_setup import sigint_shutdown_ray
from utility.graph import video_summary
from utility.run import Runner, evaluate
from utility.timer import Timer
from utility import pkg
from env.gym_env import create_env



def train(agent, env, eval_env, buffer):
    def initialize_rms(env, step, reset, obs, reward, **kwargs):
        agent.update_obs_rms(obs)
        agent.update_reward_rms(reward)
    def collect(env, step, reset, reward, next_obs, **kwargs):
        agent.update_reward_rms(reward)
        kwargs['reward'] = agent.normalize_reward(reward)
        buffer.add(**kwargs)

    step = agent.env_step
    action_selector = lambda *args, **kwargs: agent(*args, **kwargs, update_rms=True)
    runner = Runner(env, agent, step=step, nsteps=agent.N_STEPS)
    print('Start to initialize observation running stats...')
    runner.run(action_selector=env.random_action, 
                step_fn=initialize_rms,
                nsteps=50*agent.N_STEPS)
    runner.step = step

    to_log = Every(agent.LOG_PERIOD, agent.LOG_PERIOD)
    to_eval = Every(agent.EVAL_PERIOD)
    print('Training starts...')
    while step < agent.MAX_STEPS:
        start_env_step = agent.env_step
        start_time = time.time()
        step = runner.run(action_selector=action_selector, step_fn=collect)
        agent.store(fps=(step-start_env_step)/(time.time()-start_time))
        
        reset = runner.env_output.reset
        _, terms = agent(runner.env_output.obs, update_curr_state=False, reset=reset)
        buffer.finish(terms['value'])
        start_train_step = agent.train_step
        start_time = time.time()
        agent.learn_log(step)
        agent.store(tps=(agent.train_step-start_train_step)/(time.time()-start_time))
        buffer.reset()

        if to_eval(agent.train_step):
            with TempStore(agent.get_states, agent.reset_states):
                scores, epslens, video = evaluate(
                    eval_env, agent, record=agent.RECORD, size=(64, 64))
                if agent.RECORD:
                    video_summary(f'{agent.name}/sim', video, step=step)
                agent.store(eval_score=scores, eval_epslen=epslens)
        if to_log(agent.train_step) and 'score' in agent._logger:
            agent.log(step)
            agent.save()

def main(env_config, model_config, agent_config, buffer_config):
    algo = agent_config['algorithm']

    create_model, Agent = pkg.import_agent(config=agent_config)
    PPOBuffer = pkg.import_module('buffer', algo=algo).PPOBuffer

    silence_tf_logs()
    configure_gpu()
    configure_precision(agent_config['precision'])

    use_ray = env_config.get('n_workers', 1) > 1
    if use_ray:
        ray.init()
        sigint_shutdown_ray()

    env = create_env(env_config, force_envvec=True)
    eval_env_config = env_config.copy()
    eval_env_config['seed'] += 1000
    eval_env_config['n_workers'] = 1
    eval_env_config['n_envs'] = 8
    eval_env_config.pop('reward_clip', False)
    eval_env_config.pop('life_done', False)
    eval_env = create_env(eval_env_config, force_envvec=True)

    buffer_config['n_envs'] = env.n_envs
    buffer = PPOBuffer(buffer_config)

    models = create_model(model_config, env)
    
    agent = Agent(name=env.name, 
                config=agent_config, 
                models=models, 
                dataset=buffer,
                env=env)

    agent.save_config(dict(
        env=env_config,
        model=model_config,
        agent=agent_config,
        buffer=buffer_config
    ))

    train(agent, env, eval_env, buffer)

    if use_ray:
        ray.shutdown()
