# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NFSP agents trained on Kuhn Poker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability, policy_aggregator
import os, sys
import numpy as np
from jax import nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.elements.builder import ElementsBuilder
from core.log import setup_logging
from core.utils import *
from core.typing import ModelPath, get_basic_model_name, dict2AttrDict
from core.ckpt.pickle import set_weights_for_agent
from env.func import create_env
from run.args import parse_eval_args
from run.utils import search_for_all_configs, search_for_config


class RLPolicy(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, agent, aid):
    player_ids = [0, 1]
    super().__init__(env.game, player_ids)
    self.env = env
    self.aid = aid
    self._agent = agent
    self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    assert cur_player == self.aid == player_id, (
      cur_player, self.aid, player_id
    )
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
      state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
      observations=self._obs, 
      rewards=None, 
      discounts=None, 
      step_type=None
    )

    obs = self.env.get_obs(info_state)
    obs['prev_action'] = np.zeros(legal_actions, dtype=np.float32)
    obs['prev_reward'] = np.zeros((), dtype=np.float32)
    for k, v in obs.items():
      obs[k] = np.expand_dims(np.expand_dims(v, 0), 0)

    _, terms, _ = self._agent.actor(obs)
    probs = np.squeeze(nn.softmax(terms['mu_logits']))
    prob_dict = {action: probs[action] for action in legal_actions}
    return prob_dict


class JointPolicy(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, agents):
    player_ids = [0, 1]
    super().__init__(env.game, player_ids)
    self.env = env
    self._agents = agents
    self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
      state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
      observations=self._obs, 
      rewards=None, 
      discounts=None, 
      step_type=None
    )

    obs = self.env.get_obs(info_state)
    obs['prev_action'] = np.zeros(legal_actions, dtype=np.float32)
    obs['prev_reward'] = np.zeros((), dtype=np.float32)
    for k, v in obs.items():
      obs[k] = np.expand_dims(np.expand_dims(v, 0), 0)

    _, terms, _ = self._agents[cur_player].actor(obs)
    probs = np.squeeze(nn.softmax(terms['mu_logits']))
    prob_dict = {action: probs[action] for action in legal_actions}
    return prob_dict

def build_agent(builder, config, env):
  name = 'params'
  model = ModelPath(config['root_dir'], config['model_name'])
  agent = builder.build_acting_agent_from_scratch(
    config, 
    env_stats=env.stats(),
    build_monitor=False, 
    to_build_for_eval=False, 
    to_restore=False
  ).agent
  i = 0
  while True:
    try:
      # Exception happens when the file is written by the training process
      set_weights_for_agent(agent, model, name=name)
      break
    except Exception as e:
      print('Set weights failed:', e)
      import time
      time.sleep(5)
      i += 1
      if i > 10:
        raise e
  return agent

def get_latest_configs(configs):
  if len(configs) == 2:
    return configs
  latest_configs = [None, None]
  iteration1 = -1
  iteration2 = -1
  for config in configs:
    if config.aid == 0:
      if config.iteration > iteration1:
        iteration1 = config.iteration
        latest_configs[0] = config
    else:
      if config.iteration > iteration2:
        iteration2 = config.iteration
        latest_configs[1] = config
  return latest_configs

def build_latest_joint_policy(configs, env):
  builder = ElementsBuilder(configs[0], env.stats())
  agent1 = build_agent(builder, configs[0], env)
  agent2 = build_agent(builder, configs[1], env)
  joint_policy = JointPolicy(env, [agent1, agent2])
  return joint_policy

def build_policies(configs, env):
  policies = [[], []]
  builder = ElementsBuilder(configs[0], env.stats())
  for config in configs:
    agent = build_agent(builder, config, env)
    policies[config.aid].append(RLPolicy(
      env, agent, config.aid))
  probs = [np.ones(len(pls)) for pls in policies]
  pol_agg = policy_aggregator.PolicyAggregator(env.game)
  aggr_policy = pol_agg.aggregate([0, 1], policies, probs)
  return aggr_policy


def main(
  configs, 
  step, 
  filename=None, 
  avg=True, 
  latest=True, 
  write_to_disk=True, 
):
  configs = [dict2AttrDict(c) for c in configs]
  for config in configs:
    config.runner.n_runners = 1
    config.env.n_runners = 1
    config.env.n_envs = 1
  env = create_env(config.env)

  set_seed(config.seed)
  configure_gpu(None)

  nash_conv = {'step': step}

  if avg:
    aggr_policy = build_policies(configs, env)
    br1 = exploitability.best_response(env.game, aggr_policy, 0)
    br2 = exploitability.best_response(env.game, aggr_policy, 1)
    nash_conv.update(dict(
      nash_conv=(br1['nash_conv'] + br2['nash_conv']) / 2, 
      nash_conv1=br1['nash_conv'], 
      nash_conv2=br2['nash_conv'], 
      expl2=br1['best_response_value'], 
      expl1=br2['best_response_value'], 
      expl=(br1['best_response_value'] + br2['best_response_value']) / 2, 
      on_policy_value1=br1['on_policy_value'], 
      on_policy_value2=br2['on_policy_value'], 
    ))

  latest_configs = get_latest_configs(configs)
  if latest:
    joint_policy = build_latest_joint_policy(latest_configs, env)
    br1 = exploitability.best_response(env.game, joint_policy, 0)
    br2 = exploitability.best_response(env.game, joint_policy, 1)
    nash_conv.update(dict(
      latest_nash_conv=(br1['nash_conv'] + br2['nash_conv']) / 2, 
      latest_nash_conv1=br1['nash_conv'], 
      latest_nash_conv2=br2['nash_conv'], 
      latest_expl2=br1['best_response_value'], 
      latest_expl1=br2['best_response_value'], 
      latest_expl=(br1['best_response_value'] + br2['best_response_value']) / 2, 
      latest_on_policy_value1=br1['on_policy_value'], 
      latest_on_policy_value2=br2['on_policy_value'], 
    ))

  if write_to_disk:
    root_dir = latest_configs[0].root_dir
    model_name = get_basic_model_name(latest_configs[0].model_name)
    if filename is not None:
      with open(filename, 'a') as f:
        if os.stat(filename).st_size == 0:
          f.write('\t'.join(['root_dir', 'model_name', *nash_conv.keys()]) + '\n')
        f.write('\t'.join([
          root_dir, 
          model_name, 
          *[f'{v:.3f}' if isinstance(v, float) else f'{v}' for v in nash_conv.values()]
        ]) + '\n')

  return nash_conv


if __name__ == '__main__':
  args = parse_eval_args()

  setup_logging(args.verbose)
  # load respective config
  if len(args.directory) == 1:
    configs = search_for_all_configs(args.directory[0])
  else:
    configs = [search_for_config(d) for d in args.directory]
  config = configs[0]

  nash_conv = main(configs, 0)
  print(nash_conv)
