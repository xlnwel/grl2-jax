import time
import ray

from utility.ray_setup import sigint_shutdown_ray
from utility.yaml_op import load_config
from env.gym_env import create_env
from replay.func import create_replay_center
from run import pkg


def main(env_config, model_config, agent_config, replay_config):
    if 'BipedalWalkerHardcore' in env_config['name']:
        algo = agent_config['algorithm']
        root_dir = agent_config['root_dir']
        model_name = agent_config['model_name']
        directory = pkg.get_package(algo, 0, '/')
        config = load_config(f'{directory}/bwh_sac_config.yaml')
        env_config = config['env']
        model_config = config['model']
        agent_config = config['agent']
        replay_config = config['replay']
        agent_config['root_dir'] = root_dir
        agent_config['model_name'] = model_name

    if 'atari' in env_config['name'] or 'dmc' in env_config['name']:
        ray.init(num_cpus=12, num_gpus=1)
        n_cpus = 4
    else:
        ray.init(num_cpus=6, num_gpus=1, 
            memory=8*1024**3, object_store_memory=7*1024**3)
        n_cpus = 2
    
    sigint_shutdown_ray()

    replay = create_replay_center(replay_config)

    model_fn, Agent = pkg.import_agent(agent_config)
    am = pkg.import_module('actor', config=agent_config)
    fm = pkg.import_module('func', config=agent_config)

    name = agent_config['algorithm']
    Learner = am.get_learner_class(Agent)
    learner = fm.create_learner(
        Learner=Learner, 
        model_fn=model_fn, 
        replay=replay, 
        config=agent_config, 
        model_config=model_config, 
        env_config=env_config,
        replay_config=replay_config,
        n_cpus=n_cpus)
   
    Worker = am.get_worker_class()
    workers = []
    pids = []
    for wid in range(agent_config['n_workers']):
        worker = fm.create_worker(
            Worker=Worker, 
            worker_id=wid, 
            model_fn=model_fn, 
            config=agent_config, 
            model_config=model_config, 
            env_config=env_config, 
            buffer_config=replay_config)
        pids.append(worker.run.remote(learner, replay))
        workers.append(worker)

    Evaluator = am.get_evaluator_class()
    evaluator = fm.create_evaluator(
        Evaluator=Evaluator,
        model_fn=model_fn,
        config=agent_config,
        model_config=model_config,
        env_config=env_config)
    evaluator.run.remote(learner)
    while not ray.get(replay.good_to_learn.remote()):
        time.sleep(1)

    learner.start_learning.remote()

    ray.get(pids)

    ray.get(learner.save.remote())
    
    ray.shutdown()
