import numpy as np
import gym
import env.football.gfootball.env as football_env

from env.grf_env.selected_agents import SelectedAgents
from env.utils import *

def do_flatten(obj):
    """Run flatten on either python list or numpy array."""
    if type(obj) == list:
        return np.array(obj).flatten()
    return obj.flatten()

class Representation:
    RAW='raw'
    CUSTOM='custom'
    MAT='mat'
    SIMPLE115='simple115v2'


class GRF:
    def __init__(
        self,
        # built-in configs for grf
        env_name,
        representation=Representation.SIMPLE115,
        rewards='scoring,checkpoints',
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        render=False,
        write_video=False,
        dump_frequency=1000,
        logdir='results/grf',
        extra_players=None,
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0,
        # custom grf configs
        shared_ckpt_reward=False, 
        shared_reward=False, 
        shared_policy=False, 
        score_reward_scale=None, 
        # required configs for grl
        max_episode_steps=3000,
        use_action_mask=False,
        use_sample_mask=False, 
        uid2aid=None,
        seed=None, 
        use_idx=False, 
        use_hidden=False, 
        use_event=False, 
        agentwise_global_state=False, 
        selected_agents=False, 
        **kwargs,
    ):
        self.name = env_name
        self.representation = representation
        self.to_render = render
        self.shared_reward = shared_reward
        self.score_reward_scale = score_reward_scale
        self.selected_agents = selected_agents

        # assert number_of_left_players_agent_controls in (1, 11), \
        #     number_of_left_players_agent_controls
        # assert number_of_right_players_agent_controls in (0, 1, 11), \
        #     number_of_right_players_agent_controls

        if uid2aid is None:
            if shared_policy:
                if number_of_left_players_agent_controls > 0:
                    uid2aid = tuple(np.zeros(number_of_left_players_agent_controls, dtype=np.int32)) \
                        + tuple(np.ones(number_of_right_players_agent_controls, dtype=np.int32))
                else:
                    uid2aid = tuple(np.zeros(number_of_right_players_agent_controls, dtype=np.int32))
            else:
                if number_of_left_players_agent_controls > 0:
                    uid2aid = tuple(np.arange(number_of_left_players_agent_controls + number_of_right_players_agent_controls, dtype=np.int32))
                else:
                    uid2aid = tuple(np.arange(number_of_right_players_agent_controls, dtype=np.int32))
        self.uid2aid = uid2aid
        self.aid2uids = compute_aid2uids(self.uid2aid)
        self.n_units = len(self.uid2aid)
        self.n_agents = len(self.aid2uids)

        if representation != Representation.SIMPLE115:
            representation = Representation.RAW
            if self.representation == Representation.CUSTOM:
                # assert number_of_left_players_agent_controls in (1, 11), number_of_left_players_agent_controls
                # assert number_of_right_players_agent_controls in (0, 11), number_of_right_players_agent_controls
                from env.grf_env.custom_obs import FeatureEncoder
                self.feat_encoder = FeatureEncoder(
                    self.aid2uids, 
                    use_idx=use_idx, 
                    use_hidden=use_hidden, 
                    use_event=use_event, 
                    use_action_mask=use_action_mask, 
                    agentwise_global_state=agentwise_global_state
                )
            elif self.representation == Representation.MAT:
                from env.grf_env.mat_obs import FeatureEncoder
                self.feat_encoder = FeatureEncoder()
            elif self.representation == Representation.RAW:
                pass
            else:
                raise NotImplementedError(f'Unknown representation {self.representation}')
        else:
            self.feat_encoder = lambda x: x

        self.number_of_left_players_agent_controls = number_of_left_players_agent_controls
        self.number_of_right_players_agent_controls = number_of_right_players_agent_controls

        assert number_of_left_players_agent_controls + number_of_right_players_agent_controls \
            == self.n_units, \
                (self.uid2aid, number_of_left_players_agent_controls, number_of_right_players_agent_controls)

        self.shared_ckpt_reward = shared_ckpt_reward
        if self.shared_ckpt_reward and rewards == 'scoring,checkpoints':
            rewards = 'scoring'
        other_config_options = {} if seed is None else {'seed': seed}
        # print('other config options', other_config_options)
        if selected_agents:
            self.env = SelectedAgents(
                self.name, 
                representation=representation,
                rewards=rewards,
                write_goal_dumps=write_goal_dumps,
                write_full_episode_dumps=write_full_episode_dumps,
                render=render,
                write_video=write_video,
                dump_frequency=dump_frequency,
                logdir=logdir,
                extra_players=extra_players,
                number_of_left_players_agent_controls=number_of_left_players_agent_controls,
                number_of_right_players_agent_controls=number_of_right_players_agent_controls,
            )
        else:
            self.env = football_env.create_environment(
                self.name, 
                representation=representation,
                rewards=rewards,
                write_goal_dumps=write_goal_dumps,
                write_full_episode_dumps=write_full_episode_dumps,
                render=render,
                write_video=write_video,
                dump_frequency=dump_frequency,
                logdir=logdir,
                extra_players=extra_players,
                number_of_left_players_agent_controls=number_of_left_players_agent_controls,
                number_of_right_players_agent_controls=number_of_right_players_agent_controls,
                other_config_options=other_config_options, 
            )

        self.max_episode_steps = max_episode_steps

        self.use_action_mask = use_action_mask  # if action mask is used
        self.use_sample_mask = use_sample_mask              # if life mask is used
        self.use_idx = use_idx
        self.use_hidden = use_hidden
        self.use_event = use_event

        self.action_space = [
            self.env.action_space
            if isinstance(self.env.action_space, gym.spaces.MultiDiscrete) 
            else self.env.action_space 
            for _ in range(self.n_agents)]
        self.action_shape = [() for _ in self.action_space]
        self.action_dim = [19 for a in self.env.action_space.nvec]
        self.action_dtype = [np.int32 for _ in self.action_space]
        self.is_action_discrete = [True for _ in self.action_space]

        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        obs = self.reset()
        self.obs_shape = [{k: v.shape[-1:] for k, v in o.items()} for o in obs]
        self.obs_dtype = [{k: v.dtype for k, v in o.items()} for o in obs]

        # The following stats should be updated in self.step and be reset in self.reset
        # The episodic score we use to evaluate agent's performance. It excludes shaped rewards
        self._score = np.zeros(self.n_units, dtype=np.float32)
        # The accumulated episodic rewards we give to the agent. It includes shaped rewards
        self._dense_score = np.zeros(self.n_units, dtype=np.float32)
        # The length of the episode
        self._epslen = 0
        self._left_score = np.zeros(self.n_units, dtype=np.float32)
        self._right_score = np.zeros(self.n_units, dtype=np.float32)

        self._prev_action = [-1 for _ in self.action_dim]
        self._consecutive_action = np.zeros(self.n_units, bool)

        self._checkpoint_reward = .1
        self._num_checkpoints = 10
        self._collected_checkpoints = [0, 0]

    def _get_observation_shape(self):
        if self.representation == Representation.CUSTOM:
            return self.feat_encoder.get_obs_shape(self.n_agents)
        elif self.representation == Representation.MAT:
            obs_shape = self.feat_encoder.obs_shape
        else:
            obs_shape = self.env.observation_space.shape \
                if self.n_units == 1 else self.env.observation_space.shape[1:]
        shape = []
        for i in range(self.n_agents):
            s = dict(
                obs=obs_shape, 
                global_state=obs_shape, 
            )
            if self.use_action_mask:
                s['action_mask'] = (self.action_space[i].n,)
            if self.use_idx:
                s['idx'] = (self.n_units,)
            if self.use_hidden:
                s['hidden_state'] = obs_shape
            if self.use_event:
                s['event'] = (3,)
            shape.append(s)

        return shape

    def _get_observation_dtype(self):
        if self.representation == Representation.CUSTOM:
            return self.feat_encoder.get_obs_dtype(self.n_agents)
        obs_dtype = np.float32

        dtype = []
        for _ in range(self.n_agents):
            d = dict(
                obs=obs_dtype, 
                global_state=obs_dtype, 
            )
            if self.use_action_mask:
                d['action_mask'] = bool
            if self.use_idx:
                d['idx'] = np.float32
            if self.use_hidden:
                d['hidden_state'] = obs_dtype
            if self.use_event:
                d['event'] = np.float32
            dtype.append(d)

        return dtype

    def random_action(self):
        action = np.concatenate([
            np.random.randint(0, self.action_dim[0], len(uids)) 
            for uids in self.aid2uids
        ])

        return action

    def reset(self):
        obs = self.env.reset()

        self._score = np.zeros(self.n_units, dtype=np.float32)
        self._dense_score = np.zeros(self.n_units, dtype=np.float32)
        self._epslen = 0
        self._left_score = np.zeros(self.n_units, dtype=np.float32)
        self._right_score = np.zeros(self.n_units, dtype=np.float32)
        self._ckpt_score = np.zeros(self.n_units, dtype=np.float32)

        self._prev_action = [-1 for _ in self.action_dim]
        self._consecutive_action = np.zeros(self.n_units, bool)
        self._collected_checkpoints = [0, 0]

        return self._get_obs(obs)

    def step(self, action):
        action_oh = [np.zeros((len(uids), a), np.float32) 
            for uids, a in zip(self.aid2uids, self.action_dim)]
        for uids, a, oh in zip(self.aid2uids, action, action_oh):
            oh[np.arange(len(uids)), a] = 1
        obs, reward, done, info = self.env.step(action)

        reward = self._get_reward(reward, info)
        if self.number_of_right_players_agent_controls == 0:
            self._ckpt_score += reward - info['score_reward']
        else:
            self._ckpt_score[:self.number_of_left_players_agent_controls] += \
                reward[:self.number_of_left_players_agent_controls] - info['score_reward']
            self._ckpt_score[self.number_of_left_players_agent_controls:] += \
                reward[self.number_of_left_players_agent_controls:] + info['score_reward']

        if self.shared_reward:
            if self.number_of_right_players_agent_controls == 0:
                rewards = np.ones(self.n_units, dtype=np.float32) * np.sum(reward)
            else:
                rewards = np.zeros(self.n_units, dtype=np.float32)
                rewards[:self.number_of_left_players_agent_controls] = \
                    np.sum(reward[:self.number_of_left_players_agent_controls])
                rewards[self.number_of_left_players_agent_controls:] = \
                    np.sum(reward[self.number_of_left_players_agent_controls:])
        else:
            rewards = np.reshape(reward, -1)

        self._epslen += 1
        self._dense_score += rewards
        self._left_score += 1 if info['score_reward'] == 1 else 0
        self._right_score += 1 if info['score_reward'] == -1 else 0
        diff_score = self._left_score - self._right_score
        if self.name.startswith('11_vs_11') and self._epslen == self.max_episode_steps:
            done = True
            self._score = np.where(
                self._left_score < self._right_score, -1, 
                self._left_score > self._right_score)
            self._score[self.number_of_left_players_agent_controls:] = \
                - self._score[self.number_of_left_players_agent_controls:]
        else:
            self._score = diff_score > 0
        dones = np.tile(done, self.n_units)

        self._consecutive_action = np.array(
            [pa == a for pa, a in zip(self._prev_action, action)], bool)
        self._prev_action = action
        if self.number_of_right_players_agent_controls != 0:
            diff_score[-self.number_of_right_players_agent_controls:] *= -1
        info = {
            'score': self._score,
            'dense_score': self._dense_score,
            'left_score': self._left_score,
            'right_score': self._right_score,
            'diff_score': diff_score,
            'win_score': diff_score > 0,
            # 'non_loss_score': diff_score >= 0,
            # 'consecutive_action': self._consecutive_action, 
            'checkpoint_score': self._ckpt_score, 
            'epslen': self._epslen,
            'game_over': done
        }

        agent_rewards = [np.reshape(rewards[uids], -1) for uids in self.aid2uids]
        agent_dones = [np.reshape(dones[uids], -1) for uids in self.aid2uids]
        agent_obs = self._get_obs(obs, action_oh, agent_rewards)

        return agent_obs, agent_rewards, agent_dones, info

    def render(self):
        if not self.to_render:
            self.env.render(mode='rgb_array')
            self.to_render = True
        obs = self._raw_obs()[0]
        return obs['frame']

    def close(self):
        return self.env.close()

    def _get_obs(self, obs, action=None, reward=None):
        if action is None:
            action = [np.zeros((len(uids), a), np.float32) 
                for uids, a in zip(self.aid2uids, self.action_dim)]
        if reward is None:
            reward = [np.zeros(len(uids), np.float32) for uids in self.aid2uids]

        if self.representation == Representation.CUSTOM:
            agent_obs = self.feat_encoder.construct_observations(
                obs, action, reward)
        elif self.representation == Representation.MAT:
            obs_array = []
            act_masks = []
            for i, o in enumerate(obs):
                o = self.feat_encoder(o, i)
                act_masks.append(o['avail'])
                o = np.concatenate([
                    np.array(v, np.float32).flatten() for v in o.values()
                ])
                obs_array.append(o)
            obs = np.stack(obs_array)
            act_masks = np.stack(act_masks).astype(bool)
            agent_obs = [dict(
                obs=obs[uids], 
                global_state=obs[uids], 
                action_mask=act_masks[uids], 
            ) for aid, uids in enumerate(self.aid2uids)]
        else:
            if self.n_units == 1:
                obs = np.expand_dims(obs, 0)
            agent_obs = [dict(
                obs=obs[uids], 
                global_state=obs[uids], 
            ) for aid, uids in enumerate(self.aid2uids)]
            if self.use_idx:
                for o, uids in zip(agent_obs, self.aid2uids):
                    o['idx'] = np.eye(len(uids), dtype=np.float32)
            if self.use_event:
                event = self._get_event()
                for o, uids in zip(agent_obs, self.aid2uids):
                    o['event'] = event[uids]
            if self.use_hidden:
                for o in agent_obs:
                    o['hidden_state'] = o['global_state']

        return agent_obs

    def _get_event(self):
        observations = self._raw_obs()
        events = []
        for aid, uids in enumerate(self.aid2uids):
            for u in uids:
                e = observations[u]['ball_owned_team']
                events.append(np.zeros(3, np.float32))
                events[-1][e] = 1

        return np.stack(events)

    def _get_reward(self, reward, info):
        def add_ckpt_reward(reward, side):
            assert side in [0, -1], side
            if (info['score_reward'] == 1 and side == 0) or \
                    (info['score_reward'] == -1 and side == -1):
                reward += self._checkpoint_reward * (
                    self._num_checkpoints - self._collected_checkpoints[side])
                self._collected_checkpoints[side] = self._num_checkpoints
            else:
                o = self.env.unwrapped.observation()[side]
                if 'ball_owned_team' not in o or o['ball_owned_team'] != side:
                    return reward
                d = ((o['ball'][0] - 1) ** 2 + o['ball'][1] ** 2) ** 0.5
                while self._collected_checkpoints[side] < self._num_checkpoints:
                    threshold = (.99 - .8 / (self._num_checkpoints - 1)
                        * self._collected_checkpoints[side])
                    if d > threshold:
                        break
                    reward += self._checkpoint_reward
                    self._collected_checkpoints[side] += 1
            return reward

        if self.shared_ckpt_reward:
            if info['score_reward'] != 0 and self.score_reward_scale is not None:
                reward = self.score_reward_scale * info['score_reward'] * np.ones_like(reward)
            if self.number_of_right_players_agent_controls == 0:
                reward = add_ckpt_reward(reward, 0)
            else:
                reward[:self.number_of_left_players_agent_controls] = add_ckpt_reward(
                    reward[:self.number_of_left_players_agent_controls], 0
                )
                reward[self.number_of_left_players_agent_controls:] = add_ckpt_reward(
                    reward[self.number_of_left_players_agent_controls:], -1
                )

        return reward

    def _raw_obs(self):
        obs = self.env.unwrapped.observation()
        if self.selected_agents:
            obs = [obs[i] for i in self.env.controlled_players]
        return obs

    def seed(self, seed):
        return seed

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', '-l', type=int, default=4)
    parser.add_argument('--right', '-r', type=int, default=0)
    parser.add_argument('--step', '-s', type=int, default=10)
    parser.add_argument('--unit', '-u', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = {
        'env_name': 'academy_counterattack_hard',
        'representation': 'simple115v2',
        'rewards': 'scoring,checkpoints', 
        'number_of_left_players_agent_controls': args.left,
        'number_of_right_players_agent_controls': args.right,
        'shared_ckpt_reward': True, 
        'use_action_mask':True, 
        'uid2aid': None,
        'use_idx': True,
        'use_hidden': False, 
        'agentwise_global_state': False, 
        'render': False, 
        'selected_agents': True, 
        'seed': 1
    }

    from tools.display import print_dict_info, print_dict
    from tools.utils import batch_dicts
    env = GRF(**config)
    left = args.left
    obs_left = []
    obs_right = []
    obs = env.reset()
    # obs_left.append(obs[0])
    # obs_right.append(obs[1])
    obs = env._raw_obs()
    ids = np.array([o['active'] for o in obs])
    # print(obs[0]['obs'].reshape(-1, 5))
    print(ids)
    print('ball_owned_team', [o['ball_owned_team'] for o in obs])
    # ids = np.array([o['obs'][0, -13+i] for i, o in enumerate(obs)])
    for i in range(args.step):
        a = env.random_action()
        obs, rew, done, info = env.step(a)
        obs = env._raw_obs()
        new_ids = np.array([o['active'] for o in obs])
        # new_ids = np.array([o['obs'][0, -13+i] for i, o in enumerate(obs)])
        np.testing.assert_equal(ids, new_ids)
        print('ball_owned_team', [o['ball_owned_team'] for o in obs])
        # print(obs[0]['obs'].reshape(-1, 5))
        if np.all(done):
            print('Done ball_owned_team', [o['ball_owned_team'] for o in obs])
            env.reset()
            new_ids = np.array([o['active'] for o in env._raw_obs()])
            # new_ids = np.array([o['obs'][0, -13+i] for i, o in enumerate(obs)])
            np.testing.assert_equal(ids, new_ids)
    # o = []
    # o.extend(do_flatten(env._raw_obs()[0]['left_team']))
    # for k, v in env._raw_obs()[0].items():
    #     print(k, v)
    # print('obs event', obs[0]['event'], obs[0]['initial_event'])
    # random.seed(0)
    # np.random.seed(0)
    # shift = 0
    # for i in range(args.step):
    #     a = env.random_action()
    #     o, r, d, info = env.step(a)
    #     print('obs event', o[0]['event'], o[0]['initial_event'])
    #     idx = np.where(o[0]['obs'][0] != o[0]['obs'][1])
    #     if np.any(d):
    #         env.reset()

    # raw_env = football_env.create_environment(
    #     config['env_name'], 
    #     representation=config['representation'],
    #     rewards=config['rewards'],
    #     number_of_left_players_agent_controls=args.left,
    #     number_of_right_players_agent_controls=args.right,
    # )
    # obs = raw_env.reset()
    # for i, o in enumerate(obs):
    #     print(i, 'active', o['active'])
    # for k, v in o.items():
    #     if isinstance(v, np.ndarray):
    #         print(k, v.shape)
    #     else:
    #         print(k, v)
