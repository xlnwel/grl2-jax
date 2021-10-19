import functools
import signal
import sys
import numpy as np

from core.dataset import create_dataset
from core.monitor import create_monitor
from core.tf_config import configure_gpu, configure_precision, silence_tf_logs
from core.utils import save_config
from utility.utils import TempStore
from utility.run import Runner, evaluate
from utility.timer import Every, Timer
from utility import pkg
from env.func import create_env


def train(agent, env, eval_env, buffer):
    collect_fn = pkg.import_module('elements.utils', algo=agent.name).collect
    collect = functools.partial(collect_fn, buffer)

    suite_name = env.name.split("_")[0] \
        if '_' in env.name else 'builtin'
    em = pkg.import_module(suite_name, pkg='env')
    info_func = em.info_func if hasattr(em, 'info_func') else None

    step = agent.get_env_step()
    runner = Runner(env, agent, step=step, nsteps=agent.N_STEPS, info_func=info_func)

    def initialize_rms():
        print('Start to initialize running stats...')
        for _ in range(10):
            runner.run(action_selector=env.random_action, step_fn=collect)
            agent.actor.update_obs_rms(np.concatenate(buffer['obs']))
            discount = np.logical_and(buffer['discount'], 1 - buffer['reset'])
            agent.actor.update_reward_rms(buffer['reward'], discount)
            buffer.reset()
        buffer.clear()
        agent.set_env_step(runner.step)
        agent.save(print_terminal_info=True)

    if step == 0 and agent.actor.is_obs_normalized:
        initialize_rms()

    runner.step = step
    # print("Initial running stats:", *[f'{k:.4g}' for k in agent.get_rms_stats() if k])
    to_record = Every(agent.LOG_PERIOD, agent.LOG_PERIOD)
    to_eval = Every(agent.EVAL_PERIOD)
    rt = Timer('run')
    tt = Timer('train')
    et = Timer('eval')
    lt = Timer('log')

    def evaluate_agent(step, eval_env, agent):
        if eval_env is not None:
            with TempStore(agent.model.get_states, agent.model.reset_states):
                with et:
                    eval_score, eval_epslen, video = evaluate(
                        eval_env, agent, n=agent.N_EVAL_EPISODES, 
                        record_video=agent.RECORD_VIDEO, size=(64, 64))
                if agent.RECORD_VIDEO:
                    agent.video_summary(video, step=step)
                agent.store(
                    eval_score=eval_score, 
                    eval_epslen=eval_epslen)

    def record_stats(step):
        with lt:
            agent.store(**{
                'misc/train_step': agent.get_train_step(),
                'time/run': rt.total(), 
                'time/train': tt.total(),
                'time/eval': et.total(),
                'time/log': lt.total(),
                'time/run_mean': rt.average(), 
                'time/train_mean': tt.average(),
                'time/eval_mean': et.average(),
                'time/log_mean': lt.average(),
            })
            agent.record(step=step)
            agent.save()

    print('Training starts...')
    while step < agent.MAX_STEPS:
        start_env_step = agent.get_env_step()
        with rt:
            step = runner.run(step_fn=collect)
        # NOTE: normalizing rewards here may introduce some inconsistency 
        # if normalized rewards is fed as an input to the network.
        # One can reconcile this by moving normalization to collect 
        # or feeding the network with unnormalized rewards.
        # The latter is adopted in our implementation. 
        # However, the following line currently doesn't store
        # a copy of unnormalized rewards
        discount = np.logical_and(buffer['discount'], 1 - buffer['reset'])
        agent.actor.update_reward_rms(buffer['reward'], discount)
        buffer.update('reward', agent.actor.normalize_reward(buffer['reward']), field='all')
        agent.record_inputs_to_vf(runner.env_output)
        value = agent.compute_value()
        buffer.finish(value)

        start_train_step = agent.get_train_step()
        with tt:
            agent.train_record()
        agent.store(
            fps=(step-start_env_step)/rt.last(),
            tps=(agent.get_train_step()-start_train_step)/tt.last())
        agent.set_env_step(step)
        buffer.reset()

        if to_eval(agent.get_train_step()) or step > agent.MAX_STEPS:
            evaluate_agent(step, eval_env, agent)

        if to_record(agent.get_train_step()) and agent.contains_stats('score'):
            record_stats(step)

def main(config, train=train):
    silence_tf_logs()
    configure_gpu()
    configure_precision(config.agent['precision'])

    use_ray = config.env.get('n_workers', 1) > 1
    if use_ray:
        import ray
        from utility.ray_setup import sigint_shutdown_ray
        ray.init()
        sigint_shutdown_ray()

    root_dir = config.agent.root_dir
    model_name = config.agent.model_name
    name = config.agent.algorithm

    def build_envs():
        env = create_env(config.env, force_envvec=True)
        eval_env_config = config.env.copy()
        if config.env.pop('do_evaluation', True):
            if 'num_levels' in eval_env_config:
                eval_env_config['num_levels'] = 0
            if 'seed' in eval_env_config:
                eval_env_config['seed'] += 1000
            eval_env_config['n_workers'] = 1
            for k in list(eval_env_config.keys()):
                # pop reward hacks
                if 'reward' in k:
                    eval_env_config.pop(k)
            
            eval_env = create_env(eval_env_config, force_envvec=True)
        else: 
            eval_env = None
        
        return env, eval_env
    
    env, eval_env = build_envs()

    def sigint_handler(sig, frame):
        signal.signal(sig, signal.SIG_IGN)
        env.close()
        if eval_env is not None:
            eval_env.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, sigint_handler)

    env_stats = env.stats()
    def build_elements():
        create_model = pkg.import_module(
            name='elements.model', algo=name, place=-1).create_model
        create_loss = pkg.import_module(
            name='elements.loss', algo=name, place=-1).create_loss
        create_trainer = pkg.import_module(
            name='elements.trainer', algo=name, place=-1).create_trainer
        create_actor = pkg.import_module(
            name='elements.actor', algo=name, place=-1).create_actor

        model = create_model(config.model, env_stats)
        loss = create_loss(config.loss, model)
        trainer = create_trainer(config.trainer, model, loss, env_stats)
        actor = create_actor(config.actor, model)
        
        return model, trainer, actor
    model, trainer, actor = build_elements()

    def build_buffer_dataset():
        config.buffer['n_envs'] = env.n_envs
        config.buffer['state_keys'] = model.state_keys
        config.buffer['use_dataset'] = config.buffer.get('use_dataset', False)
        create_buffer = pkg.import_module(
            'elements.buffer', config=config.agent).create_buffer
        buffer = create_buffer(config.buffer)
        
        if config.buffer['use_dataset']:
            am = pkg.import_module('elements.utils', config=config.agent)
            data_format = am.get_data_format(
                config.trainer, env_stats, model)
            dataset = create_dataset(buffer, env, 
                data_format=data_format, one_hot_action=False)
        else:
            dataset = buffer

        return buffer, dataset
    buffer, dataset = build_buffer_dataset()

    def build_strategy():
        create_strategy = pkg.import_module(
            'elements.strategy', config=config.agent).create_strategy
        strategy = create_strategy(
            name, config.strategy, model, trainer, actor, dataset)
        
        return strategy
    strategy = build_strategy()

    def build_agent():
        create_agent = pkg.import_module(
            'elements.agent', config=config.agent).create_agent
        monitor = create_monitor(root_dir, model_name, name)

        agent = create_agent(
            config=config.agent, 
            strategy=strategy, 
            monitor=monitor,
            name=name
        )

        return agent
    agent = build_agent()
    save_config(root_dir, model_name, config)

    train(agent, env, eval_env, buffer)

    if use_ray:
        env.close()
        if eval_env is not None:
            eval_env.close()
        ray.shutdown()
