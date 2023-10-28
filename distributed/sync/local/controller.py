from math import ceil
import logging
import cloudpickle
from functools import partial
import time
from typing import List, Tuple, Union
import numpy as np
import ray

from .agent_manager import AgentManager
from .runner_manager import RunnerManager
from ..remote.monitor import Monitor
from ..remote.parameter_server import ParameterServer
from ..remote.parameter_server_sp import SPParameterServer
from core.ckpt.base import YAMLCheckpointBase
from core.log import do_logging
from core.typing import ModelPath, get_basic_model_name, AttrDict, dict2AttrDict
from core.utils import save_code
from env.func import get_env_stats
from game.alpharank import AlphaRank
from run.utils import search_for_all_configs, search_for_config
from tools.process import run_ray_process
from tools.schedule import PiecewiseSchedule
from tools.timer import Every, Timer, timeit
from tools.utils import batch_dicts, eval_config, modify_config, prefix_name
from tools import yaml_op


logger = logging.getLogger(__name__)
timeit = partial(timeit, period=1)


def _check_configs_consistency(
  configs: List[AttrDict], 
  keys: List[str], 
):
  for key in keys:
    for i, c in enumerate(configs):
      c[key].pop('aid')
      for k in c[key].keys():
        if k != 'root_dir' and k != 'seed':
          assert configs[0][key][k] == c[key][k], (
            key, i, k, c[key][k], configs[0][key][k])


def _setup_configs(
  configs: List[AttrDict], 
  env_stats: List[AttrDict]
):
  configs = [dict2AttrDict(c, to_copy=True) for c in configs]
  for aid, config in enumerate(configs):
    config.aid = aid
    config.buffer.n_envs = env_stats.n_envs
    root_dir = config.root_dir
    model_name = '/'.join([config.model_name, f'a{aid}'])
    modify_config(
      config, 
      root_dir=root_dir, 
      model_name=model_name, 
      aid=aid
    )

  return configs


def _compute_pbt_steps(n_runners, n_steps, n_online_runners, n_agent_runners):
  worker_steps = n_runners * n_steps
  n_agent_runners = n_online_runners + n_agent_runners
  n_pbt_steps = ceil(worker_steps / n_agent_runners)
  assert n_agent_runners * n_pbt_steps >= worker_steps, (n_agent_runners, n_pbt_steps)

  return n_pbt_steps


class Controller(YAMLCheckpointBase):
  def __init__(
    self, 
    config: AttrDict,
    name='controller',
    to_restore=True,
  ):
    self.config = eval_config(config.controller)
    self._model_path = ModelPath(
      self.config.root_dir, 
      get_basic_model_name(self.config.model_name)
    )
    self._dir = '/'.join(self._model_path)
    self._path = f'{self._dir}/{name}.yaml'
    do_logging(self._model_path)
    save_code(self._model_path)

    self._iteration = 1
    self._steps = 0
    self._pids = []

    if to_restore:
      self.restore()

  """ Manager Building """
  def build_managers_for_evaluation(self, config: AttrDict):
    if config.self_play:
      ParameterServerCls = SPParameterServer
    else:
      ParameterServerCls = ParameterServer
    self.parameter_server: ParameterServerCls = \
      ParameterServerCls.as_remote().remote(
        config=config.asdict(),
        to_restore_params=False, 
      )

    self.runner_manager: RunnerManager = RunnerManager(
      ray_config=config.ray_config.runner,
      parameter_server=self.parameter_server,
      monitor=None
    )

    self.runner_manager.build_runners(config, evaluation=True)

  def build_managers(self, configs: List[AttrDict]):
    configs = [dict2AttrDict(c) for c in configs]
    _check_configs_consistency(configs, [
      'controller', 
      'parameter_server', 
      'ray_config', 
      'monitor', 
      'runner', 
      'env', 
    ])

    do_logging('Retrieving Environment Stats...', logger=logger)
    env_stats = get_env_stats(configs[0].env)
    self.suite, _ = configs[0].env.env_name.split('-', 1)
    self.configs = _setup_configs(configs, env_stats)

    config = configs[0]

    self.n_runners = config.runner.n_runners
    self.n_steps = config.runner.n_steps
    self.n_envs = config.env.n_envs
    self.steps_per_run = self.n_runners * self.n_envs * self.n_steps

    if config.self_play:
      ParameterServerCls = SPParameterServer
    else:
      ParameterServerCls = ParameterServer
    do_logging('Building Parameter Server...', logger=logger)
    self.parameter_server: ParameterServerCls = \
      ParameterServerCls.as_remote().remote(
        config=config.asdict(),
        to_restore_params=True, 
      )
    ray.get(self.parameter_server.build.remote(
      configs=[c.asdict() for c in self.configs],
      env_stats=env_stats.asdict(),
    ))

    do_logging('Buiding Monitor...', logger=logger)
    self.monitor: Monitor = Monitor.as_remote().remote(
      config.asdict(), 
      self.parameter_server
    )

    do_logging('Building Agent Manager...', logger=logger)
    self.agent_manager: AgentManager = AgentManager(
      ray_config=config.ray_config.agent,
      env_stats=env_stats, 
      parameter_server=self.parameter_server,
      monitor=self.monitor
    )

    do_logging('Building Runner Manager...', logger=logger)
    self.runner_manager: RunnerManager = RunnerManager(
      ray_config=config.ray_config.runner,
      parameter_server=self.parameter_server,
      monitor=self.monitor
    )

  """ Training """
  @timeit
  def pbt_train(self):
    iteration_step_scheduler = self._get_iteration_step_scheduler(
      self.config.max_version_iterations, 
      self.config.max_steps_per_iteration
    )

    while self._iteration <= self.config.max_version_iterations: 
      do_logging(f'Starting Iteration {self._iteration}', logger=logger)
      self.initialize_actors()
      max_steps = iteration_step_scheduler(self._iteration)
      ray.get(self.monitor.set_max_steps.remote(max_steps))

      self.train(self.agent_manager, self.runner_manager, max_steps)

      self.cleanup()
  
    do_logging(f'Training Finished. Total Iterations: {self._iteration-1}', logger=logger)
    self._log_remote_stats(self._pids, record=True)

  def _get_iteration_step_scheduler(
    self, 
    max_version_iterations: Union[int, List, Tuple], 
    max_steps_per_iteration: int
  ):
    if isinstance(max_steps_per_iteration, (List, Tuple)):
      iteration_step_scheduler = PiecewiseSchedule(max_steps_per_iteration)
    else:
      iteration_step_scheduler = PiecewiseSchedule([(
        max_version_iterations, 
        max_steps_per_iteration
      )])
    return iteration_step_scheduler

  @timeit
  def initialize_actors(self):
    model_weights, is_raw_strategy = ray.get(
      self.parameter_server.sample_training_strategies.remote(self._iteration))
    self.current_models = [m.model for m in model_weights]
    do_logging(
      f'Training Strategies at Iteration {self._iteration}: {self.current_models}', 
      logger=logger)

    configs = self._prepare_configs(self.n_runners, self.n_steps, self._iteration)

    self.agent_manager.build_agents(configs)
    self.agent_manager.set_model_weights(model_weights, wait=True)
    self.agent_manager.publish_weights(wait=True)

    self.active_models = [model for model, _ in model_weights]
    self.active_configs = [
      search_for_config(model, to_attrdict=False) 
      for model in self.active_models
    ]

    self.runner_manager.build_runners(
      configs, 
      remote_buffers=self.agent_manager.get_agents(),
      active_models=self.active_models, 
    )
    do_logging(f'Finish Building Runners', logger=logger)
    self._initialize_rms(self.active_models, is_raw_strategy)

  @timeit
  def train(
    self, 
    agent_manager: AgentManager, 
    runner_manager: RunnerManager, 
    max_steps: int
  ):
    agent_manager.start_training()
    to_restart_runners = Every(
      self.config.restart_runners_period, 
      0 if self.config.restart_runners_period is None \
        else self._steps + self.config.restart_runners_period
    )
    to_eval = Every(self.config.eval_period, final=max_steps)
    to_store = Every(self.config.store_period, final=max_steps)
    eval_pids = []
    while self._steps < max_steps:
      model_weights = self._retrieve_model_weights()

      steps = sum(runner_manager.run_with_model_weights(model_weights))
      self._steps += self.steps_per_run   # adding a fixed number of steps gives nicer logging stats for plotting
      # self._steps += steps
      # do_logging(f'finishing sampling: total steps={self._steps}, steps={steps}')

      self._post_processing(to_eval, to_restart_runners, to_store, eval_pids)

    ready_pids, eval_pids = ray.wait(eval_pids)
    self._log_remote_stats_for_models(
      ready_pids, self.active_models, step=self._steps)

    self._finish_iteration()

  @timeit
  def cleanup(self):
    do_logging(f'Cleaning up for Training Iteration {self._iteration}...', logger=logger)
    oids = [
      self.parameter_server.archive_training_strategies.remote(),
      self.monitor.clear_iteration_stats.remote()
    ]
    self.runner_manager.destroy_runners()
    self.agent_manager.destroy_agents()
    self._iteration += 1
    self.save()
    ray.get(oids)

  """ Implementation for <pbt_train> """
  def _prepare_configs(self, n_runners: int, n_steps: int, iteration: int):
    configs = [dict2AttrDict(c, to_copy=True) for c in self.configs]
    runner_stats = ray.get(self.parameter_server.get_runner_stats.remote())
    assert self._iteration == runner_stats.iteration, (self._iteration, runner_stats.iteration)
    n_online_runners = runner_stats.n_online_runners
    n_agent_runners = runner_stats.n_agent_runners
    n_pbt_steps = _compute_pbt_steps(
      n_runners, 
      n_steps, 
      n_online_runners, 
      n_agent_runners, 
    )
    for c in configs:
      c.trainer.n_runners = n_online_runners + n_agent_runners
      c.buffer.n_runners = n_online_runners + n_agent_runners
      c.trainer.n_steps = n_pbt_steps
      c.runner.n_steps = n_pbt_steps
      c.buffer.n_steps = n_pbt_steps
    runner_stats.n_pbt_steps = n_pbt_steps
    do_logging(runner_stats, prefix=f'Runner Stats at Iteration {self._iteration}', 
      logger=logger)
    self._log_stats(runner_stats, self._iteration)
    return configs

  def _initialize_rms(self, models: List[ModelPath], is_raw_strategy: List[bool]):
    if self.config.initialize_rms and any(is_raw_strategy):
      aids = [i for i, is_raw in enumerate(is_raw_strategy) if is_raw]
      do_logging(f'Initializing RMS for Agents: {aids}', logger=logger)
      self.runner_manager.set_current_models(models)
      self.runner_manager.random_run(aids)

  """ Implementation for <train> """
  def _retrieve_model_weights(self):
    model_weights = ray.get(self.parameter_server.get_strategies.remote())
    while model_weights is None:
      time.sleep(.025)
      model_weights = ray.get(self.parameter_server.get_strategies.remote())
    assert len(model_weights) == self.n_runners, (len(model_weights), self.n_runners)
    return model_weights

  def _post_processing(
    self, 
    to_eval: Every, 
    to_restart_runners: Every, 
    to_store: Every, 
    eval_pids: List[ray.ObjectRef]
  ):
    if to_eval(self._steps):
      pid = self._eval(self._steps)
      if pid:
        eval_pids.append(pid)
    if to_restart_runners(self._steps):
      do_logging('Restarting Runners', logger=logger)
      self.runner_manager.destroy_runners()
      self.runner_manager.build_runners(
        self.configs, 
        remote_buffers=self.agent_manager.get_agents(),
        active_models=self.active_models, 
      )
    if self._pids:
      ready_pids, self._pids = ray.wait(self._pids, timeout=1e-5)
      self._log_remote_stats(ready_pids, record=True)
    if eval_pids:
      ready_pids, eval_pids = ray.wait(eval_pids, timeout=1)
      self._log_remote_stats_for_models(
        ready_pids, self.active_models, step=self._steps)
    if to_store(self._steps):
      self.monitor.save_all.remote(self._steps)
      self.save()

  def _finish_iteration(self):
    if self.suite == 'spiel':
      do_logging('Computing Nash Convergence...', logger=logger)
      self._pids.append(self.compute_nash_conv(
        self._iteration, avg=True, 
        latest=True, write_to_disk=False
      ))
    do_logging(f'Finishing Iteration {self._iteration}')
    ray.get([
      self.monitor.save_all.remote(self._steps),
      self.monitor.save_payoff_table.remote(self._iteration)
    ])
    self._steps = 0

  """ Statistics Logging """
  def _log_stats(self, stats: dict, step: int, record: bool=False):
    self.monitor.store_stats.remote(
      stats=stats, 
      step=step, 
      record=record
    )

  def _log_remote_stats(
    self, 
    pids: List[ray.ObjectRef], 
    step: int=None, 
    record: bool=False
  ):
    if pids:
      stats_list = ray.get(pids)
      for stats in stats_list:
        if step is None:
          step = stats.pop('step')
      stats = batch_dicts(stats_list)
    else:
      stats = {}
    self.monitor.store_stats.remote(
      stats=stats, 
      step=step, 
      record=record
    )
  def _log_remote_stats_for_models(
    self, 
    pids: List[ray.ObjectRef], 
    models: List[ModelPath], 
    record: bool=False, 
    step: int=None
  ):
    if pids:
      stats_list = ray.get(pids)
      stats = batch_dicts(stats_list)
      stats = prefix_name(stats, name='eval')
    else:
      stats = {}
    for m in models:
      self.monitor.store_stats_for_model.remote(
        m, stats, step=step, record=record)

  """ Evaluation """
  @timeit
  def evaluate_all(self, total_episodes, filename):
    ray.get(self.parameter_server.reset_payoffs.remote(from_scratch=False))
    self.runner_manager.evaluate_all(total_episodes)
    payoffs = self.parameter_server.get_payoffs.remote()
    counts = self.parameter_server.get_counts.remote()
    payoffs = ray.get(payoffs)
    counts = ray.get(counts)
    do_logging('Final Payoffs', logger=logger)
    for i in range(len(payoffs)):
      print(payoffs[i])
    do_logging(f'Final Counts: {np.mean(counts)}, {np.std(counts)}', logger=logger)
    alpha_rank = AlphaRank(1000, 5, 0)
    ranks, mass = alpha_rank.compute_rank(payoffs, return_mass=True)
    print('Alpha Rank Results:\n', ranks)
    print('Mass at Stationary Point:\n', mass)
    path = f'{self._dir}/{filename}.pkl'
    with open(path, 'wb') as f:
      cloudpickle.dump((payoffs, counts), f)
    do_logging(f'Payoffs have been saved at {path}', logger=logger)

  def _eval(self, step):
    if self.suite == 'spiel':
      pid = self.compute_nash_conv(
        step, avg=True, latest=True, 
        write_to_disk=False, configs=self.active_configs
      )
    else:
      pid = None
    return pid

  def compute_nash_conv(self, step, avg, latest, write_to_disk, configs=None):
    if configs is None:
      configs = search_for_all_configs(self._dir, to_attrdict=False)
    from run.spiel_eval import main
    filename = '/'.join([self._dir, 'nash_conv.txt'])
    pid = run_ray_process(
      main, 
      configs, 
      step=step, 
      filename=filename,
      avg=avg, 
      latest=latest, 
      write_to_disk=write_to_disk
    )
    return pid

  """ Checkpoints """
  def save(self):
    yaml_op.dump(self._path, iteration=self._iteration, steps=self._steps)
