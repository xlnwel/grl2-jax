import time
import numpy as np
import ray 

from core.tf_config import configure_gpu, configure_precision, silence_tf_logs
from utility.utils import Every, TempStore
from utility.ray_setup import sigint_shutdown_ray
from utility.graph import video_summary, image_summary
from utility.run import Runner, evaluate
from utility.timer import Timer
from utility import pkg
# from algo.rnd.env import make_env
from env.func import create_env


def train(agent, env, eval_env, buffer):
    obs_list = []
    def collect_obs(env, step, reset, obs, **kwargs):
        obs_list.append(obs)
        if len(obs) >= 100:
            obs = np.stack(obs_list, axis=1)
            agent.update_obs_rms(obs)
            obs_list[:] = []

    def collect(env, step, reset, next_obs, **kwargs):
        buffer.add(**kwargs)

    step = agent.env_step
    runner = Runner(env, agent, step=step, nsteps=agent.N_STEPS)
    if not agent.rnd_rms_restored():
        print('Start to initialize observation running stats...')
        runner.run(action_selector=env.random_action, 
                    step_fn=collect_obs, 
                    nsteps=50*agent.N_STEPS)
        if obs_list:
            obs = np.stack(obs_list, axis=1)
            agent.update_obs_rms(obs)
        agent.save()
        runner.step = step
    del obs_list

    # print("Initial running stats:", *[
    #     k.shape if isinstance(k, np.ndarray) else f'{k:.4g}' 
    #     for k in agent.get_running_stats() if k])
    to_log = Every(agent.LOG_PERIOD, agent.LOG_PERIOD)
    to_eval = Every(agent.EVAL_PERIOD)
    print('Training starts...')
    while step < agent.MAX_STEPS:
        start_env_step = agent.env_step
        with Timer('env') as et:
            step = runner.run(step_fn=collect)
        agent.store(fps=(step-start_env_step)/et.last())

        agent.record_last_env_output(runner.env_output)
        value_int, value_ext = agent.compute_value()
        obs = buffer.get_obs(runner.env_output.obs)
        assert obs.shape[:2] == (env.n_envs, agent.N_STEPS+1)
        assert obs.dtype == np.uint8
        agent.update_obs_rms(obs)
        norm_obs = agent.normalize_obs(obs)
        reward_int = agent.compute_int_reward(norm_obs[:, 1:])
        buffer.finish(reward_int, norm_obs[:, :-1], value_int, value_ext)
        agent.store(
            reward_int_max=np.max(reward_int),
            reward_int_min=np.min(reward_int),
            reward_int=np.mean(reward_int),
            reward_int_std=np.std(reward_int),
            )

        start_train_step = agent.train_step
        with Timer('train') as tt:
            agent.learn_log(step)
        agent.store(tps=(agent.train_step-start_train_step)/tt.last())
        buffer.reset()

        if to_eval(agent.train_step):
            with TempStore(agent.get_states, agent.reset_states):
                scores, epslens, video = evaluate(
                    eval_env, agent, 
                    record=True,
                    video_len=4500)
                video_summary(f'{agent.name}/sim', video, step=step)
                if eval_env.n_envs == 1:
                    rews_int, rews_ext = agent.retrieve_eval_rewards()
                    assert len(rews_ext) == len(rews_int) == video.shape[1], (len(rews_ext), len(rews_int), video.shape[1])
                    n = 10
                    idxes_int = rews_int.argsort()[::-1][:n]
                    idxes_ext = rews_ext.argsort()[::-1][:n]
                    assert idxes_int.shape == idxes_ext.shape, (idxes_int.shape, idxes_ext.shape)
                    
                    imgs_int = video[0, idxes_int]
                    imgs_ext = video[0, idxes_ext]
                    rews_int = rews_int[idxes_int]
                    rews_ext = rews_ext[idxes_ext]
                    terms = {
                        **{f'eval/reward_int_{i}': rews_int[i] for i in range(0, n)},
                        **{f'eval/reward_ext_{i}': rews_ext[i] for i in range(0, n)},
                    }
                    agent.store(**terms)
                    imgs = np.concatenate([imgs_int[:n], imgs_ext[:n]], 0)
                    image_summary(f'{agent.name}/img', imgs, step=step)

                    # info = eval_env.info()[0]
                    # episode = info.get('episode', {'visited_rooms': 1})
                    # agent.store(visited_rooms_max=len(episode['visited_rooms']))
                    agent.histogram_summary(
                        {'eval/action': agent.retrieve_eval_actions()}, step=step)   
                agent.store(eval_score=scores, eval_epslen=epslens)

        if to_log(agent.train_step) and agent.contains_stats('score'):
            agent.store(
                episodes=runner.episodes,
                train_step=agent.train_step,
            )
            agent.log(step)
            agent.save()

def main(env_config, model_config, agent_config, buffer_config):
    algo = agent_config['algorithm']
    env = env_config['name']
    if 'atari' not in env:
        print('Any changes to config is dropped as we switch to a non-atari environment')
        from utility import yaml_op
        root_dir = agent_config['root_dir']
        model_name = agent_config['model_name']
        directory = pkg.get_package(algo, 0, '/')
        config = yaml_op.load_config(f'{directory}/config2.yaml')
        env_config = config['env']
        model_config = config['model']
        agent_config = config['agent']
        buffer_config = config['buffer']
        agent_config['root_dir'] = root_dir
        agent_config['model_name'] = model_name
        env_config['name'] = env

    create_model, Agent = pkg.import_agent(config=agent_config)
    Buffer = pkg.import_module('buffer', algo=algo).Buffer

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
    eval_env_config['n_envs'] = 1
    eval_env_config['reward_clips'] = 0
    eval_env = create_env(eval_env_config, force_envvec=True)

    buffer_config['n_envs'] = env.n_envs
    buffer = Buffer(buffer_config)

    models = create_model(model_config, env)
    
    agent = Agent(name='ppo', 
                config=agent_config, 
                models=models, 
                dataset=buffer,
                env=env)
    # restore RMSs
    agent.save_config(dict(
        env=env_config,
        model=model_config,
        agent=agent_config,
        buffer=buffer_config
    ))

    train(agent, env, eval_env, buffer)

    if use_ray:
        ray.shutdown()
