# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.elements.builder import ElementsBuilder
from core.log import setup_logging
from core.tf_config import *
from core.typing import ModelPath, get_aid
from core.ckpt.pickle import set_weights_for_agent
from env.func import create_env
from run.args import parse_eval_args
from run.utils import search_for_all_configs, search_for_config


class FPEMPolicies(policy.Policy):
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
        probs = np.squeeze(terms['pi'])
        prob_dict = {action: probs[action] for action in legal_actions}
        return prob_dict

def build_policies(configs, env):
    policies = [[], []]
    builder = ElementsBuilder(configs[0], env.stats())
    for config in configs:
        elements = builder.build_acting_agent_from_scratch(
            config, 
            env_stats=env.stats(),
            build_monitor=False, 
            to_build_for_eval=False, 
            to_restore=False
        )
        model = ModelPath(config['root_dir'], config['model_name'])
        print(model)
        aid = get_aid(model.model_name)
        assert aid == config.aid, (aid, config.aid)
        filename = 'params.pkl'
        set_weights_for_agent(elements.agent, model, filename=filename)
        policies[config.aid].append(FPEMPolicies(
            env, elements.agent, config.aid))

    return policies


def main(configs, result_file=None):
    for config in configs:
        config.runner.n_runners = 1
        config.env.n_runners = 1
        config.env.n_envs = 1
    env = create_env(config.env)

    silence_tf_logs()
    configure_gpu()
    configure_precision(config.precision)

    policies = build_policies(configs, env)
    probs = [np.ones(len(pls)) for pls in policies]
    pol_agg = policy_aggregator.PolicyAggregator(env.game)
    aggr_policy = pol_agg.aggregate([0, 1], policies, probs)

    expl = exploitability.nash_conv(env.game, aggr_policy, False)
    print(expl)

    if result_file is not None:
        with open(result_file, 'a') as f:
            f.write(f'{expl}\n')


if __name__ == '__main__':
    args = parse_eval_args()

    setup_logging(args.verbose)

    # load respective config
    if len(args.directory) == 1:
        configs = search_for_all_configs(args.directory[0])
    else:
        configs = [search_for_config(d) for d in args.directory]
    config = configs[0]

    main(configs)
