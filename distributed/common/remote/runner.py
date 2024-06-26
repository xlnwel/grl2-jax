from typing import Any, Dict, List, Set, Tuple, Union, Callable
import collections
import threading
import numpy as np
import ray

from .parameter_server import ParameterServer
from .monitor import Monitor
from core.ckpt.pickle import set_weights_for_agent
from core.elements.agent import Agent
from core.elements.buffer import Buffer
from core.elements.builder import ElementsBuilder
from core.mixin.actor import RMS
from core.names import ANCILLARY, MODEL, TRAIN_STEP
from distributed.common.remote.base import RayBase
from core.typing import ModelStats, ModelWeights, ModelPath, dict2AttrDict
from env.func import create_env
from env.typing import EnvOutput
from env.utils import divide_env_output
from tools.timer import Timer, timeit


SEED_MULTIPLIER = 1000


class MultiAgentRunner(RayBase):
  def __init__(
    self, 
    runner_id, 
    configs: Union[List[dict], dict], 
    evaluation: bool, 
    parameter_server: ParameterServer=None, 
    remote_buffers: List[RayBase]=None, 
    active_models: List[ModelPath]=None, 
    monitor: Monitor=None,
  ):
    if isinstance(configs, list):
      configs = [dict2AttrDict(c) for c in configs]
      config = configs[0]
    else:
      config = dict2AttrDict(configs)
      configs = [config]
    super().__init__(runner_id, seed=config.get('seed'))

    self.id = runner_id
    self.evaluation = evaluation
    self.algo_type = config.get("algo_type", 'onpolicy')

    env_config = self._setup_env_config(config.env)
    self.env = create_env(env_config, no_remote=True)
    self.env_stats = self.env.stats()
    self.n_envs = self.env_stats.n_envs
    self.n_agents = self.env_stats.n_agents
    self.n_units = self.env_stats.n_units
    self.uid2aid = self.env_stats.uid2aid
    self.uid2gid = self.env_stats.uid2gid
    self.aid2uids = self.env_stats.aid2uids
    self.gid2uids = self.env_stats.gid2uids
    self.aid2gids = self.env_stats.aid2gids
    self.n_units_per_agent = [len(uids) for uids in self.aid2uids]
    self.is_multi_agent = self.env_stats.is_multi_agent
    self.is_simultaneous_move = self.env_stats.is_simultaneous_move

    self.parameter_server = parameter_server
    self.remote_buffers: List[RayBase] = remote_buffers

    self.self_play = config.self_play
    assert not self.self_play or self.n_agents == 2, (self.self_play, self.n_agents == 2)
    # if self.remote_buffers is not None:
    #   assert len(self.remote_buffers) == self.n_agents, \
    #     (len(self.remote_buffers), self.n_agents)
    self.active_models: Set[ModelPath] = set(active_models) if active_models else set()
    self.current_models: List[ModelPath] = [None for _ in range(self.n_agents)]
    self.is_agent_active: List[bool] = [False for _ in range(self.n_agents)]
    self.monitor: Monitor = monitor

    self.env_output = self.env.output()
    if self.self_play:
      self.scores = []
    else:
      self.scores = [[] for _ in range(self.n_agents)]
    self.score_metric = config.parameter_server.get('score_metric', 'score')

    self.builder = ElementsBuilder(config, self.env_stats)

    self.build_from_configs(configs)
  
  def _get_local_buffer_type(self):
    return 'local' if self.is_simultaneous_move else 'tblocal'

  def build_from_configs(self, configs: Union[List[Dict], Dict]):
    if self.self_play:
      config = configs[0]
      configs = [dict2AttrDict(config) for _ in range(self.n_agents)]
    if isinstance(configs, list):
      assert len(configs) == self.n_agents, (len(configs), self.n_agents)
      configs = [dict2AttrDict(c) for c in configs]
    else:
      configs = [dict2AttrDict(configs) for _ in range(self.n_agents)]
    config = configs[0]
    self.config = config.runner
    
    self.n_steps = self.config.n_steps
    self.steps_per_run = self.n_envs * self.n_steps
    self._steps = 0
    self._total_steps = 0

    self.agents: List[Agent] = []
    self.buffers: List[Buffer] = []
    self.rms: List[RMS] = []

    for aid, config in enumerate(configs):
      config.model.seed += self.id * SEED_MULTIPLIER
      config.buffer.type = self._get_local_buffer_type()
      # update n_steps for consistency
      config.buffer.n_steps = self.n_steps
      config.buffer.sample_keys.append(TRAIN_STEP)
      elements = self.builder.build_acting_agent_from_scratch(
        config, 
        env_stats=self.env_stats, 
        build_monitor=True, 
        to_build_for_eval=self.evaluation, 
        to_restore=False
      )
      self.agents.append(elements.agent)

      self.rms.append(elements.actor.get_raw_rms())

      buffer = self.builder.build_buffer(
        elements.model, 
        config=config.buffer, 
        env_stats=self.env_stats, 
        aid=aid, 
        runner_id=self.id, 
      )
      self.buffers.append(buffer)
    assert len(self.agents) == len(self.rms) == self.n_agents, (
      len(self.agents), len(self.rms), self.n_agents)

  def get_total_steps(self):
    return self._total_steps

  """ Running Routines """
  def random_run(self, aids=None):
    """ Random run the environment to collect running stats """
    def add_env_output(data):
      agent_env_outs = divide_env_output(self.env_output)
      for d, eo in zip(data, agent_env_outs):
        for k in self.rms[0].get_obs_names():
          d[k].append(eo.obs[k])
        if 'sample_mask' in eo.obs:
          d['sample_mask'].append(eo.obs['sample_mask'])
        d['reward'].append(eo.reward)
        d['discount'].append(eo.discount)
      return data

    data = [collections.defaultdict(list) for _ in range(self.n_agents)]
    step = 0
    data = add_env_output(data)
    while step < self.n_steps:
      action = self.env.random_action()
      self.env_output = self.env.step(action)
      data = add_env_output(data)
      step += 1
    for i, d in enumerate(data):
      for k, v in d.items():
        d[k] = np.array(v)
      self._update_rms_from_batch(i, dict2AttrDict(d))

    if aids is None:
      aids = range(self.n_agents)
    for aid in aids:
      self._send_aux_stats(aid)

  def start_running(self):
    self.run_signal = True
    self._running_thread = threading.Thread(target=self.run_loop, daemon=True)
    self._running_thread.start()

  def stop_running(self):
    self.run_signal = False
    self._running_thread.join()

  def run_loop(self):
    while self.run_signal:
      mids = ray.get(self.parameter_server.get_prepared_strategies.remote(self.id))
      self.run_with_model_weights(mids)

  def run_with_model_weights(self, mids: List[ModelWeights]):
    @timeit
    def set_strategies(mids):
      for aid, mid in enumerate(mids):
        model_weights = ray.get(mid)
        self.is_agent_active[aid] = model_weights.model in self.active_models
        self.current_models[aid] = model_weights.model
        assert set(model_weights.weights).issubset(set([MODEL, ANCILLARY, TRAIN_STEP])) or set(model_weights.weights) == set(['aid', 'iid', 'path']), set(model_weights.weights)
        self.agents[aid].set_strategy(model_weights, env=self.env)
        # do_logging(f'Runner {self.id} receives weights of train step {model_weights.weights["train_step"]}')
      if self.self_play:
        self.is_agent_active = [True, False]
      assert any(self.is_agent_active), (self.active_models, self.current_models)

    def send_stats(steps, n_episodes):
      for aid, is_active in enumerate(self.is_agent_active):
        if is_active:
          self._send_aux_stats(aid)
          if n_episodes > 0:
            self._send_run_stats(aid, steps, n_episodes)
      self._update_payoffs()

    def stop_fn(step, **kwargs):
      return step >= self.n_steps
    
    set_strategies(mids)
    steps, n_episodes = self.run(stop_fn=stop_fn)
    self._steps += self.steps_per_run

    self._save_time_recordings()
    send_stats(self._steps, n_episodes)

    if n_episodes > 0:
      self._steps = 0
    self._total_steps += steps

    return steps

  @timeit
  def run(self, stop_fn: Callable, to_store_data: bool=True):
    for b in self.buffers:
      assert b.is_empty(), b.size()

    if self.is_simultaneous_move:
      run_func = self._run_impl_ma
    else:
      run_func = self._run_impl_ma_tb
    steps, n_episodes = run_func(stop_fn, to_store_data=to_store_data)

    return steps, n_episodes

  def evaluate(self, total_episodes, model_paths: List[ModelPath]=None, wait=False):
    def stop_fn(n_episodes, **kwargs):
      return n_episodes >= total_episodes

    if model_paths is not None:
      self.set_weights_from_model_paths(model_paths)
    steps, n_eps = self.run(stop_fn, to_store_data=False)
    pid = self._update_payoffs()
    if wait:
      ray.get(pid)

    return steps, n_eps

  def evaluate_and_return_stats(self, total_episodes=None):
    def stop_fn(n_episodes, **kwargs):
      return n_episodes > total_episodes

    steps, n_eps = self.run(stop_fn, to_store_data=False)

    stats = {
      'score': np.stack([a.get_raw_item('score') for a in self.agents], 1),
      'dense_score': np.stack([a.get_raw_item('dense_score') for a in self.agents], 1),
    }

    return steps, n_eps, stats

  """ Running Setups """
  def set_active_models(self, model_paths: List[ModelPath]):
    assert len(model_paths) == len(self.active_models), (model_paths, self.active_models)
    self.active_models = set(model_paths)

  def set_current_models(self, model_paths: List[ModelPath]):
    if len(model_paths) == 1:
      if self.self_play:
        self.current_models = [model_paths[0] for _ in self.current_models]
      else:
        self.current_models = model_paths
    else:
      assert len(model_paths) == len(self.current_models), (model_paths, self.current_models)
      self.current_models = model_paths

  def set_weights_from_configs(self, configs: List[dict], name='params'):
    assert len(configs) == len(self.current_models) == self.n_agents, (configs, self.current_models)
    for aid, (config, agent) in enumerate(zip(configs, self.agents)):
      model = ModelPath(config['root_dir'], config['model_name'])
      set_weights_for_agent(agent, model, name=name)
      self.current_models[aid] = model

  def set_weights_from_model_paths(self, model_paths: List[ModelPath], name='params'):
    assert len(model_paths) == len(self.current_models) == self.n_agents, (model_paths, self.current_models)
    for aid, (model, agent) in enumerate(zip(model_paths, self.agents)):
      set_weights_for_agent(agent, model, name=name)
      self.current_models[aid] = model

  def set_running_steps(self, n_steps):
    self.n_steps = n_steps

  """ Implementations """
  def _reset_local_buffers(self):
    [b.reset() for b in self.buffers]

  def _reset(self):
    self.env_output = self.env.reset()
    self._reset_local_buffers()
    return self.env_output

  def _run_impl_ma(self, stop_fn, to_store_data: bool=False):
    @timeit
    def agents_infer(agents: List[Agent], agent_env_outs: List[EnvOutput]):
      assert len(agent_env_outs)  == len(agents), (len(agent_env_outs), len(agents))
      action, terms = [], []
      for aid in range(self.n_agents):
        agent = agents[aid]
        env_out = agent_env_outs[aid]
        a, t = agent(env_out, evaluation=self.evaluation)
        action.append(a)
        terms.append(t)
      return action, terms

    @timeit
    def step_env(actions: List):
      self.env_output = self.env.step(actions)
      agent_env_outs = divide_env_output(self.env_output)
      return agent_env_outs
    
    @timeit
    def store_data(
      agent_env_outs: List[EnvOutput], 
      agent_actions: List, 
      agent_terms: List[dict], 
      next_agent_env_outs: List[EnvOutput]
    ):
      if to_store_data:
        assert len(agent_env_outs) == len(agent_actions) \
          == len(agent_terms) == len(next_agent_env_outs) \
          == len(self.buffers), (
            len(agent_env_outs), len(agent_actions), 
            len(agent_terms), len(next_agent_env_outs),
            len(self.buffers)
          )
        next_obs = self.env.prev_obs()
        for aid in range(self.n_agents):
          agent = self.agents[aid]
          env_out = agent_env_outs[aid]
          next_env_out = next_agent_env_outs[aid]
          buffer = self.buffers[aid]
          if self.is_agent_active[aid]:
            stats = {
              'obs': env_out.obs, 
              'action': agent_actions[aid], 
              'reward': agent.actor.normalize_reward(next_env_out.reward),
              'discount': next_env_out.discount,
              'reset': next_env_out.reset,
              'next_obs': next_obs[aid],
              TRAIN_STEP: agent.get_train_step() * np.ones_like(env_out.reset)
            }
            stats.update(agent_terms[aid])
            buffer.collect(**stats)

    def send_data(env_outs: List[Tuple]):
      if to_store_data:
        sent = False
        prev_env_output = self.env.prev_output()
        prev_agent_env_output = divide_env_output(prev_env_output)
        for aid in range(self.n_agents):
          agent = self.agents[aid]
          out = env_outs[aid]
          buffer = self.buffers[aid]
          if self.is_agent_active[aid]:
            assert buffer.is_full(), len(buffer)
            if self.algo_type == 'onpolicy':
              value = agent.compute_value(prev_agent_env_output[aid])
              rid, data, n = buffer.retrieve_all_data({
                'value': value,
                'state_reset': out.reset
              })
            else:
              rid, data, n = buffer.retrieve_all_data()
            self._update_rms_from_batch(aid, data)
            data = self._normalize_data(agent.actor, data)
            if self.self_play:
              self.remote_buffers[0].merge_data.remote(rid, data, n)
            else:
              self.remote_buffers[aid].merge_data.remote(rid, data, n)
            sent = True
      else:
        sent = True
      return sent

    step = 0
    n_episodes = 0
    agent_env_outs = divide_env_output(self.env_output)
    while not stop_fn(step=step, n_episodes=n_episodes):
      action, terms = agents_infer(self.agents, agent_env_outs)
      next_agent_env_outs = step_env(action)
      store_data(agent_env_outs, action, terms, next_agent_env_outs)
      agent_env_outs = next_agent_env_outs
      n_episodes += self._log_for_done(agent_env_outs[0].reset)
      step += 1

    send_data(agent_env_outs)

    return step * self.n_envs, n_episodes

  def _run_impl_ma_tb(self, stop_fn, to_store_data: bool=False):
    @timeit
    def agents_infer(agents: List[Agent], agent_env_outs: List[EnvOutput]):
      assert len(agent_env_outs)  == len(agents), (len(agent_env_outs), len(agents))
      action, terms = [], []
      for aid, (agent, o) in enumerate(zip(agents, agent_env_outs)):
        if len(o.obs) == 0:
          action.append([])
          terms.append([])
          continue
        a, t = agent(o, evaluation=self.evaluation)
        action.append(a)
        terms.append(t)
      return action, terms

    @timeit
    def step_env(agent_env_outs: List, agent_actions: List):
      actions = {}
      n = 0
      for out, acts in zip(agent_env_outs, agent_actions):
        if len(out.obs) == 0:
          assert len(acts) == 0, acts
          continue
        for k in acts.keys():
          if k not in actions:
            actions[k] = np.zeros((self.n_envs, 1), dtype=np.int32)
          assert len(out.obs['uid']) == len(acts[k]), (out.obs['uid'], acts[k])
          for i, eid in enumerate(out.obs['eid']):
            assert out.obs['action_mask'][i][0][acts[k][i]], (out.obs['action_mask'][i][0], acts[k][i])
            actions[k][eid] = acts[k][i]
            n += 1
      assert n == self.n_envs, (n, actions)
      self.env_output = self.env.step([actions])
      assert len(self.env_output) == self.n_agents, len(self.env_output)
      return self.env_output
    
    def store_data(
      agent_env_outs: List[EnvOutput], 
      agent_actions: List, 
      agent_terms: List[dict], 
      next_agent_env_outs: List[EnvOutput]
    ):
      if to_store_data:
        assert len(agent_env_outs) == len(agent_actions) \
          == len(agent_terms) == len(next_agent_env_outs) \
          == len(self.buffers), (
            len(agent_env_outs), len(agent_actions), 
            len(agent_terms), len(next_agent_env_outs),
            len(self.buffers)
          )

        for aid, (agent, env_out, next_env_out, buffer) in enumerate(
            zip(self.agents, agent_env_outs, next_agent_env_outs, self.buffers)):
          if self.is_agent_active[aid]:
            if len(env_out.obs) != 0:
              assert len(agent_actions[aid]) != 0, agent_actions[aid]
              stats = {
                **env_out.obs,
                'action': agent_actions[aid], 
                'reset': env_out.reset, 
                TRAIN_STEP: agent.get_train_step() * np.ones_like(env_out.reset)
              }
              stats.update(agent_terms[aid])
              buffer.add(**stats)
            else:
              assert len(agent_actions[aid]) == 0, agent_actions[aid]
            if len(next_env_out.reward) != 0:
              reward = agent.actor.normalize_reward(next_env_out.reward)
              buffer.add_reward(
                eids=next_env_out.obs['eid'], 
                uids=next_env_out.obs['uid'], 
                reward=reward,
                discount=next_env_out.discount,
              )

    @timeit
    def log_for_done(agent_env_outs):
      n_episodes = 0
      for aid, (agent, env_out) in enumerate(zip(self.agents, agent_env_outs)):
        reward = agent.actor.normalize_reward(np.array(env_out.reward))
        done_eids = []
        done_rewards = []
        for i, d in enumerate(env_out.discount):
          if np.any(d == 0):
            np.testing.assert_equal(d, 0)
            done_eids.append(env_out.obs['eid'][i])
            done_rewards.append(reward[i])
        if done_eids:
          for aid, buffer in enumerate(self.buffers):
            if self.is_agent_active[aid]:
              buffer.finish_episode(
                done_eids, 
                self.aid2uids[aid], 
                done_rewards
              )
          info = self.env.info(done_eids)
          stats = collections.defaultdict(list)
          for i in info:
            for k, v in i.items():
              stats[k].append(v)
          if self.self_play:
            self.scores += [v[self.aid2uids[1]].mean() for v in stats[self.score_metric]]
            self.agents[1].store(
              **{
                k: [vv[1] for vv in v]
                if isinstance(v[0], np.ndarray) else v
                for k, v in stats.items() if k.endswith('score')
              }
            )
          else:
            for aid, uids in enumerate(self.aid2uids):
              self.scores[aid] += [
                v[uids] for v in stats[self.score_metric]]
              self.agents[aid].store(
                **{
                  k: [vv[uids] for vv in v]
                  if isinstance(v[0], np.ndarray) else v
                  for k, v in stats.items()
                }
              )
          n_episodes += len(done_eids)
      return n_episodes

    def send_data():
      if to_store_data:
        for aid, buffer in enumerate(self.buffers):
          if self.is_agent_active[aid] and buffer.is_full():
            rid, data, n = buffer.retrieve_all_data()
            self._update_rms_from_batch(aid, data)
            data = self._normalize_data(self.agents[aid].actor, data)
            if self.self_play:
              self.remote_buffers[0].merge_data.remote(rid, data, n)
            else:
              self.remote_buffers[aid].merge_data.remote(rid, data, n)
            # assert np.all(np.any(data.action_mask, -1))

    step = 0
    n_episodes = 0
    agent_env_outs = self._reset()
    if to_store_data:
      def stop_fn(**kwargs):
        return np.all([b.is_full() for aid, b in enumerate(self.buffers) 
                       if self.is_agent_active[aid]])
    while not stop_fn(step=step, n_episodes=n_episodes):
      action, terms = agents_infer(self.agents, agent_env_outs)
      next_agent_env_outs = step_env(agent_env_outs, action)
      store_data(agent_env_outs, action, terms, next_agent_env_outs)
      agent_env_outs = next_agent_env_outs
      n_episodes += log_for_done(agent_env_outs)
      step += 1

    send_data()

    return step * self.n_envs, n_episodes

  @timeit
  def _update_rms(self, agent_env_outs: Union[EnvOutput, List[EnvOutput]]):
    assert len(self.rms) == len(agent_env_outs), (len(self.rms), len(agent_env_outs))
    for rms, out, gids in zip(self.rms, agent_env_outs, self.aid2gids):
      if len(out.obs) == 0:
        continue
      uids = [self.gid2uids[gid] for gid in gids]
      rms.update_obs_rms(
        out.obs, indices=uids, split_axis=1, mask=out.obs.sample_mask, 
        feature_mask=self.env_stats.feature_mask, axis=0)
      rms.update_reward_rms(out.reward, out.discount, axis=0)

  @timeit
  def _update_rms_from_batch(self, aid: int, data: Dict[str, Any]):
    uids = [self.gid2uids[gid] for gid in self.aid2gids[aid]]
    self.rms[aid].update_obs_rms(
      data, indices=uids, split_axis=2, mask=data.sample_mask, 
      feature_mask=self.env_stats.feature_mask)
    self.rms[aid].update_reward_rms(data.reward, data.discount, mask=data.sample_mask)

  @timeit
  def _normalize_data(self, actor, data):
    data.raw_reward = data.reward
    data.reward = actor.process_reward_with_rms(
      data.reward, data.discount, update_rms=False)
    obs = {k: data[k] for k in actor.get_obs_names()}
    data.update({f'raw_{k}': v for k, v in obs.items()})
    data = actor.normalize_obs(data, is_next=False)
    data = actor.normalize_obs(data, is_next=True)
    return data
  
  @timeit
  def _log_for_done(self, reset):
    # logging when any env is reset 
    done_env_ids = [i for i, r in enumerate(reset) if np.all(r)]
    if done_env_ids:
      info = self.env.info(done_env_ids)
      stats = collections.defaultdict(list)
      for i in info:
        for k, v in i.items():
          stats[k].append(v)
      if self.self_play:
        uids = self.aid2uids[0]
        self.scores += [
          v[uids].mean() for v in stats[self.score_metric]]
        self.agents[0].store(
          **{
            k: [vv[uids] for vv in v]
            if isinstance(v[0], np.ndarray) else v
            for k, v in stats.items()
          }
        )
      else:
        for aid, uids in enumerate(self.aid2uids):
          self.scores[aid] += [
            v[uids].mean() for v in stats[self.score_metric]]
          self.agents[aid].store(
            **{
              k: [vv[uids] for vv in v]
              if isinstance(v[0], np.ndarray) else v
              for k, v in stats.items()
            }
          )

    return len(done_env_ids)

  def _send_aux_stats(self, aid):
    aux_stats = self.rms[aid].get_rms_stats()
    self.rms[aid].reset_rms_stats()
    model = self.current_models[aid]
    assert model in self.active_models, (model, self.active_models)
    model_weights = ModelWeights(model, {ANCILLARY: aux_stats})
    self.parameter_server.update_aux_stats.remote(aid, model_weights)

  def _send_run_stats(self, aid, env_steps, n_episodes):
    stats = self.agents[aid].get_raw_stats()
    train_step = self.agents[aid].strategy.step_counter.get_train_step()
    stats['rid'] = self.id
    stats['train_steps'] = train_step
    stats['env_steps'] = env_steps
    stats['n_episodes'] = n_episodes
    model = self.current_models[aid]
    model_stats = ModelStats(model, stats)
    self.monitor.store_run_stats.remote(model_stats)

  def _setup_env_config(self, config: dict):
    config = dict2AttrDict(config)
    config.eid = config.get('eid', config.seed) + self.id + 1
    if config.get('seed') is not None:
      config.seed += self.id * SEED_MULTIPLIER
    if config.env_name.startswith('grf'):
      if self.id == 0 and self.evaluation:
        config.write_video = True
        config.dump_frequency = 1
        config.write_full_episode_dumps = True
        config.render = True
      else:
        config.write_video = False
        config.write_full_episode_dumps = False
        config.render = False
    return config

  def _update_payoffs(self):
    pid = None
    if self.self_play:
      if len(self.scores) > 0:
        pid = self.parameter_server.update_payoffs.remote(
          self.current_models, self.scores)
        self.scores = []
    else:
      if sum([len(s) for s in self.scores]) > 0:
        pid = self.parameter_server.update_payoffs.remote(
          self.current_models, self.scores)
        self.scores = [[] for _ in range(self.n_agents)]
    return pid

  def _save_time_recordings(self):
    stats = Timer.top_stats()
    for aid, is_active in enumerate(self.is_agent_active):
      if is_active:
        self.agents[aid].store(**stats)
