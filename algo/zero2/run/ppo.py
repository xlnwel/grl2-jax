import ray

from algo.zero2.remote.agent import RemoteAgent
from algo.zero2.remote.runner import RunnerManager
from env.func import get_env_stats
from utility.ray_setup import sigint_shutdown_ray
from utility.timer import Every, Timer


def main(config):
    ray.init()
    sigint_shutdown_ray()

    env_stats = get_env_stats(config.env)
    agent = RemoteAgent.as_remote(num_gpus=1).remote(config.asdict(), env_stats.asdict())
    runner_manager = RunnerManager(
        config,
        remote_agent=agent)
    root_dir = 'logs/card_gd/zero'
    model_name = 'self-play'
    runner_manager.set_other_agent_from_path(root_dir, model_name)

    train(agent, runner_manager, config.agent.LOG_PERIOD)

    ray.shutdown()


def train(
        agent: RemoteAgent, 
        runner_manager: RunnerManager, 
        LOG_PERIOD: int):
    rt = Timer('run')
    tt = Timer('train')

    train_step = ray.get(agent.get_train_step.remote())
    to_record = Every(LOG_PERIOD, train_step + LOG_PERIOD)
    step = ray.get(agent.get_env_step.remote())
    MAX_STEPS = runner_manager.max_steps()
    print('Training starts...')
    agent.start_training.remote()
    while step < MAX_STEPS:
        start_env_step = step
        with rt:
            weights = agent.get_weights.remote(opt_weights=False)
            step, stats = runner_manager.run(weights)
        
        start_train_step = train_step
        with tt:
            train_step = ray.get(agent.wait_for_train_step_update.remote())
        assert not ray.get(agent.is_buffer_ready.remote()), (start_train_step, train_step)
        agent.store.remote({
            **stats,
            **{
                'time/run': rt.total(),
                'time/run_mean': rt.average(),
                'time/train': tt.total(),
                'time/train_mean': tt.average(),
                'time/fps': (step-start_env_step)/rt.last(),
                'time/outer_tps': (train_step-start_train_step)/tt.last()}
        })

        if to_record(train_step):
            agent.record.remote(step)

    agent.record.remote(step)
