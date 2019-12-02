import ray

from utility.signal import sigint_shutdown_ray
from algo.sac.nn import SAC
from algo.sac.agent import Agent
from algo.apex_sac.learner import create_learner
from algo.apex_sac.worker import create_worker


def main(env_config, model_config, agent_config, buffer_config, restore=False, render=False):
    ray.init()
    sigint_shutdown_ray()

    learner = create_learner(
        BaseAgent=Agent,
        name='Learner', model_fn=SAC, config=agent_config, 
        model_config=model_config, env_config=env_config, 
        buffer_config=buffer_config)
    
    learner.save_config.remote(dict(
        env=env_config,
        model=model_config,
        agent=agent_config,
        buffer=buffer_config
    ))

    workers = []
    for worker_id in range(agent_config['n_workers']):
        worker = create_worker(
            name='worker', worker_id=worker_id, 
            model_fn=SAC, config=agent_config, 
            model_config=model_config, env_config=env_config, 
            buffer_config=buffer_config)
        workers.append(worker)
    
    if restore:
        ray.get(learner.restore.remote())
        ray.get([w._pull_weights.remote(learner) for w in workers])
    
    pids = ray.get([worker.run.remote(learner)
        for worker in workers])

