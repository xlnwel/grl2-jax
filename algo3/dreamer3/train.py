import time
import functools

from core.tf_config import configure_gpu, configure_precision, silence_tf_logs
from utility.graph import video_summary
from utility.tf_utils import tensor2numpy
from utility.utils import Every, TempStore
from utility.run import evaluate
from utility import pkg
from env.func import create_env
from replay.func import create_replay
from core.dataset import Dataset, process_with_env
from algo.dreamer.env import make_env


def run(env, agent, replay, step, obs=None, already_done=None, nsteps=0):
    assert env.n_envs == 1
    if agent._store_state:
        reset_terms = dict(prev_logpi=0, 
            **tensor2numpy(
                agent.rssm.get_initial_state(batch_size=env.n_envs)._asdict()))
    else:
        reset_terms = dict(prev_logpi=0)
    if obs is None:
        obs = env.reset(**reset_terms)
    if already_done is None:
        already_done = env.already_done()
    frame_skip = getattr(env, 'frame_skip', 1)
    frames_per_step = frame_skip
    nsteps = (nsteps or env.max_episode_steps) // frame_skip
    for _ in range(nsteps):
        action, terms = agent(obs, already_done, evaluation=False)
        obs, reward, done, info = env.step(action, **terms)
        already_done = info.get('already_done', False)
        step += frames_per_step
        if already_done:
            eps = info['episode']
            agent.store(score=env.score(), epslen=env.epslen())
            replay.merge(eps)
            obs = env.reset(**reset_terms)
            
    return obs, already_done, step

def train(agent, env, eval_env, replay):
    _, step = replay.count_episodes()
    step = max(agent.env_step, step)

    nsteps = agent.TRAIN_PERIOD
    obs, already_done = None, None
    while not replay.good_to_learn():
        obs, already_done, step = run(env, agent, replay, step, obs, already_done)
        
    to_log = Every(agent.LOG_PERIOD)
    to_eval = Every(agent.EVAL_PERIOD)
    print('Training starts...')
    while step < int(agent.MAX_STEPS):
        start_step = step
        start_t = time.time()
        agent.train_log(step)
        obs, already_done, step = run(
            env, agent, replay, step, obs, already_done, agent.N_UPDATES)
        duration = time.time() - start_t
        agent.store(
            fps=(step-start_step) / duration,
            tps=(agent.N_UPDATES / duration))

        if to_eval(step):
            with TempStore(agent.get_states, agent.reset_states):
                score, epslen, video = evaluate(
                    eval_env, agent, record=True, size=(64, 64))
                video_summary(f'{agent.name}/sim', video, step=step)
                agent.store(eval_score=score, eval_epslen=epslen)
            
        if to_log(step):
            agent.log(step)
            agent.save()

def main(env_config, model_config, agent_config, replay_config):
    silence_tf_logs()
    configure_gpu()
    configure_precision(env_config['precision'])

    use_ray = env_config.get('n_workers', 0) > 1
    if use_ray:
        import ray
        from utility.ray_setup import sigint_shutdown_ray
        ray.init()
        sigint_shutdown_ray()

    env = create_env(env_config, make_env)
    eval_env_config = env_config.copy()
    del eval_env_config['log_episode']
    eval_env = create_env(eval_env_config, make_env)

    create_model, Agent = pkg.import_agent(config=agent_config)
    models = create_model(model_config, env)

    agent = Agent(
        name='dreamer',
        config=agent_config,
        models=models, 
        dataset=None,
        env=env)

    replay_config['dir'] = agent_config['root_dir'].replace('logs', 'data')
    replay = create_replay(replay_config,
        state_keys=list(agent.rssm.state_size._asdict()))
    replay.load_data()
    data_format = pkg.import_module('agent', config=agent_config).get_data_format(
        env=env, 
        batch_size=agent_config['batch_size'], 
        sample_size=agent_config['sample_size'], 
        store_state=agent_config['store_state'],
        state_size=agent.rssm.state_size)
    process = functools.partial(process_with_env, 
        env=env, obs_range=[-.5, .5], one_hot_action=True)
    dataset = Dataset(replay, data_format, process)
    agent.dataset = dataset

    agent.save_config(dict(
        env=env_config,
        model=model_config,
        agent=agent_config,
        replay=replay_config
    ))

    train(agent, env, eval_env, replay)
