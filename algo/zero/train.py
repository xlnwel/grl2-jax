import collections
import functools
import signal
import sys
import numpy as np
import ray

from core.elements.builder import ElementsBuilder
from core.tf_config import configure_gpu, configure_precision, silence_tf_logs
from core.utils import save_config
from utility.utils import batch_dicts, dict2AttrDict
from utility.run import Runner, evaluate
from utility.timer import Every, Timer
from utility.typing import AttrDict
from utility import pkg
from env.func import create_env, get_env_stats


@ray.remote
class RemoteRunner:
    def __init__(self, config, name='zero', store_data=True):
        silence_tf_logs()
        configure_gpu()
        self.config = dict2AttrDict(config)
        self.store_data = store_data
        self.config.buffer.use_dataset = False
        self.config.buffer.n_envs = self.config.env.n_envs
        self.config.env.n_workers = 1

        self.env = build_env(self.config)
        env_stats = self.env.stats()

        builder = ElementsBuilder(self.config, env_stats, name=name)
        model = builder.build_model(to_build=True)
        actor = builder.build_actor(model)
        strategy = builder.build_strategy(actor=actor)
        monitor = builder.build_monitor()
        self.agent = builder.build_agent(strategy, monitor=monitor, to_save_code=False)

        if self.store_data:
            self.buffer = builder.build_buffer(model)
            collect_fn = pkg.import_module('elements.utils', algo=self.config.algorithm).collect
            self.collect = functools.partial(collect_fn, self.buffer)
        else:
            self.buffer = None
            self.collect = None
        self.runner = self.setup_runner()

    def setup_runner(self):
        suite_name = self.env.name.split("_")[0] if '_' in self.env.name else 'builtin'
        em = pkg.import_module(suite_name, pkg='env')
        info_func = em.info_func if hasattr(em, 'info_func') else None

        runner = Runner(self.env, self.agent, step=self.agent.get_env_step(), 
            nsteps=self.config.agent.N_STEPS, info_func=info_func)
        
        return runner
    
    def initialize_rms(self):
        for _ in range(10):
            self.runner.run(action_selector=self.env.random_action, step_fn=self.collect)
            self.agent.actor.update_obs_rms(np.concatenate(self.buffer['obs']))
            self.agent.actor.update_reward_rms(self.buffer['reward'], self.buffer['discount'])
            self.buffer.reset()
        self.buffer.clear()
        return self.agent.actor.get_rms_stats()

    def run(self, weights):
        if weights is not None:
            self.agent.set_weights(weights)
        step = self.runner.run(step_fn=self.collect)
        self.agent.record_inputs_to_vf(self.runner.env_output)
        value = self.agent.compute_value()
        if self.store_data:
            self.buffer.append_data({'last_value': value})
            self.buffer.stack_sequantial_memory()
            data = self.buffer.retrieve_all_data()
            return step, data, self.agent.get_raw_stats()
        else:
            return step, None, self.agent.get_raw_stats()


class RunnerManager:
    def __init__(self, config, name='zero', store_data=True):
        if isinstance(config, AttrDict):
            config = config.asdict()
        self.runners = [RemoteRunner.remote(config, name, store_data=store_data) 
            for _ in range(config['env']['n_workers'])]

    def initialize_rum(self):
        obs_rms_list, rew_rms_list = list(
            zip(*ray.get([r.initialize_rms.remote() for r in self.runners])))
        return obs_rms_list, rew_rms_list
    
    def run(self, weights):
        wid = ray.put(weights)
        steps, data, stats = list(zip(*ray.get([r.run.remote(wid) for r in self.runners])))
        stats = batch_dicts(stats, lambda x: sum(x, []))
        return steps, data, stats
    
    def evaluate(self, total_episodes, weights=None):
        n_eps = 0
        stats_list = []
        while n_eps < total_episodes:
            _, _, stats = self.run(weights)
            n_eps += len(next(iter(stats.values())))
            stats_list.append(stats)
        stats = batch_dicts(stats_list, lambda x: sum(x, []))
        return stats, n_eps


def build_env(config):
    env = create_env(config.env)
    return env


def train(agent, buffer, config):
    runner_manager = RunnerManager(config, name=agent.name)

    if agent.get_env_step() == 0 and agent.actor.is_obs_normalized:
        obs_rms_list, rew_rms_list = runner_manager.initialize_rum()
        agent.update_rms_from_stats_list(obs_rms_list, rew_rms_list)

    # print("Initial running stats:", *[f'{k:.4g}' for k in agent.get_rms_stats() if k])
    to_record = Every(agent.LOG_PERIOD, agent.LOG_PERIOD)
    # to_eval = Every(agent.EVAL_PERIOD)
    rt = Timer('run')
    tt = Timer('train')
    # et = Timer('eval')
    lt = Timer('log')

    # def evaluate_agent(step, eval_env, agent):
    #     if eval_env is not None:
    #         with TempStore(agent.model.get_states, agent.model.reset_states):
    #             with et:
    #                 eval_score, eval_epslen, video = evaluate(
    #                     eval_env, agent, n=agent.N_EVAL_EPISODES, 
    #                     record_video=agent.RECORD_VIDEO, size=(64, 64))
    #             if agent.RECORD_VIDEO:
    #                 agent.video_summary(video, step=step)
    #             agent.store(
    #                 eval_score=eval_score, 
    #                 eval_epslen=eval_epslen)

    def record_stats(step):
        with lt:
            agent.store(**{
                'misc/train_step': agent.get_train_step(),
                'time/run': rt.total(), 
                'time/train': tt.total(),
                # 'time/eval': et.total(),
                'time/log': lt.total(),
                'time/run_mean': rt.average(), 
                'time/train_mean': tt.average(),
                # 'time/eval_mean': et.average(),
                'time/log_mean': lt.average(),
            })
            agent.record(step=step)
            agent.save()

    step = agent.get_env_step()
    print('Training starts...')
    while step < agent.MAX_STEPS:
        start_env_step = agent.get_env_step()
        with rt:
            weights = agent.get_weights(opt_weights=False)
            steps, data, stats = runner_manager.run(weights)
        step = sum(steps)
        agent.set_env_step(step)

        for d in data:
            buffer.append_data(d)
        buffer.concat_batch_memory()

        # NOTE: normalizing rewards here may introduce some inconsistency 
        # if normalized rewards is fed as an input to the network.
        # One can reconcile this by moving normalization to collect 
        # or feeding the network with unnormalized rewards.
        # The latter is adopted in our implementation. 
        # However, the following line currently doesn't store
        # a copy of unnormalized rewards
        for o in agent.actor.obs_names:
            agent.actor.update_obs_rms(buffer[o], o)
        agent.actor.update_reward_rms(buffer['reward'], buffer['discount'])
        buffer.finish()

        start_train_step = agent.get_train_step()
        with tt:
            agent.train_record()
        train_step = agent.get_train_step()

        agent.store(
            **stats,
            fps=(step-start_env_step)/rt.last(),
            tps=(train_step-start_train_step)/tt.last())
        agent.set_env_step(step)
        buffer.reset()

        # if to_eval(train_step) or step > agent.MAX_STEPS:
        #     evaluate_agent(step, eval_env, agent)

        if to_record(train_step) and agent.contains_stats('score'):
            record_stats(step)

def ppo_train(config):
    import ray
    from utility.ray_setup import sigint_shutdown_ray
    ray.init()
    sigint_shutdown_ray()

    root_dir = config.agent.root_dir
    model_name = config.agent.model_name

    env_stats = get_env_stats(config.env)
    config.buffer.n_envs = env_stats.n_workers * env_stats.n_envs

    builder = ElementsBuilder(config, env_stats, name='zero')
    model = builder.build_model()
    actor = builder.build_actor(model)
    trainer = builder.build_trainer(model)
    buffer = builder.build_buffer(model)
    dataset = builder.build_dataset(buffer, model)
    strategy = builder.build_strategy(actor=actor, trainer=trainer, dataset=dataset)
    monitor = builder.build_monitor()
    agent = builder.build_agent(strategy=strategy, monitor=monitor)
 
    save_config(root_dir, model_name, builder.get_config())

    train(agent, buffer, config)

    ray.shutdown()


def bc_train(config):
    root_dir = config.agent.root_dir
    model_name = config.agent.model_name

    env_stats = get_env_stats(config.env)
    builder = ElementsBuilder(config, env_stats, name='zero')
    model = builder.build_model(to_build=True)
    actor = builder.build_actor(model)
    trainer = builder.build_trainer(model)
    buffer = builder.build_buffer(model)
    dataset = builder.build_dataset(buffer, model)
    strategy = builder.build_strategy(actor=actor, trainer=trainer, dataset=dataset)
    monitor = builder.build_monitor()
    agent = builder.build_agent(strategy=strategy, monitor=monitor)
    
    save_config(root_dir, model_name, builder.get_config())

    tt = Timer('train')
    lt = Timer('log')
    to_record = Every(agent.LOG_PERIOD, agent.LOG_PERIOD)
    def record_stats(step):
        with lt:
            agent.store(**{
                'misc/train_step': agent.get_train_step(),
                'time/train': tt.total(),
                'time/log': lt.total(),
                'time/train_mean': tt.average(),
                'time/log_mean': lt.average(),
            })
            agent.record(step=step)
            agent.save()

    step = 0
    while True:
        start_train_step = step
        with tt:
            agent.train_record()
        step = agent.get_train_step()
        agent.store(
            tps=(step-start_train_step)/tt.last())
        agent.set_env_step(step)

        if to_record(step):
            record_stats(step)

def main(config):
    silence_tf_logs()
    configure_gpu()
    configure_precision(config.precision)

    if config['training'] == 'ppo':
        ppo_train(config)
    elif config['training'] == 'bc':
        bc_train(config)
    else:
        raise ValueError(config['training'])