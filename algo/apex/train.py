import time
import ray

from utility.ray_setup import sigint_shutdown_ray
from utility.yaml_op import load_config
from utility import pkg
from env.gym_env import create_env
from replay.func import create_replay_center


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
    ray.init(num_cpus=12, num_gpus=1)
    n_cpus = 4
    
    sigint_shutdown_ray()

    replay = create_replay_center(replay_config)

    model_fn, Agent = pkg.import_agent(config=agent_config)
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
            config=agent_config, 
            model_config=model_config, 
            env_config=env_config, 
            buffer_config=replay_config,
            model_fn=model_fn, )
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

    learner.start_learning.remote()

    while ray.get(learner.is_learning.remote()):
        time.sleep(60)

    ray.get(learner.save.remote())
    
    ray.shutdown()
