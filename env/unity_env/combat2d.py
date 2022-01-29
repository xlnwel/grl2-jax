from typing import List

import numpy
import numpy as np
import gym

from .interface import UnityInterface
from .opponent import Opponent
from copy import deepcopy
import math
# from env.utils import compute_aid2uids

"""  Naming Conventions:
An Agent is the one that makes decisions, it's typically a RL agent
A Unit is a controllable unit in the environment.
Each agent may control a number of units.
All ids start from 0 and increases sequentially
"""


def compute_aid2uids(uid2aid):
    """ Compute aid2uids from uid2aid """
    aid2uids = []
    for pid, aid in enumerate(uid2aid):
        if aid > len(aid2uids):
            raise ValueError(f'uid2aid({uid2aid}) is not sorted in order')
        if aid == len(aid2uids):
            aid2uids.append((pid,))
        else:
            aid2uids[aid] += (pid,)

    return aid2uids


# NOTE: Keep the class name fixed; do not invent a new one!
# We do not rely on this for distinction!
class UnityEnv:
    def __init__(
            self,
            uid2aid,
            n_envs,
            unity_config,
            reward_config={
                'detect_reward': 0.2,
                'main_dead_reward': -10,
                'blue_dead_reward': 10
            },
            # expand kwargs for your environment
            **kwargs
    ):
        # uid2aid is a list whose indices are the unit ids and values are agent ids.
        # It specifies which agent controlls the unit.
        # We expect it to be sorted in the consecutive ascending order
        # That is, [0, 1, 1] is valid. [0, 1, 0] and [0, 0, 2] are invalid
        self.uid2aid: list = uid2aid
        self.aid2uids = compute_aid2uids(self.uid2aid)
        self.n_agents = len(self.aid2uids)  # the number of agents
        self.n_units = len(self.uid2aid)

        unity_config['n_envs'] = n_envs
        self.env = UnityInterface(**unity_config)

        self._red_names = [name for name in self.env.get_behavior_names() if name.startswith('red')]
        self._blue_names = [name for name in self.env.get_behavior_names() if name.startswith('blue')]

        # TODO sort red and blue planes(default sort will have mistakes if plane num>10)
        self._red_names.sort()
        self._blue_names.sort()
        self._red_main_ray_dim = 42
        self._red_ally_ray_dim = 35
        self._blue_ray_dim = 42
        # assert self._red_names == [
        #     'red_main?team=0', 'red_sup_0?team=0', 'red_sup_1?team=0', 'red_sup_2?team=0',
        #     'red_sup_3?team=0'
        # ]

        # the number of envs running in parallel, which must be the same as the number
        # of environment copies specified when compiing the Unity environment
        self.n_envs = n_envs
        # The same story goes with n_units
        self.n_units = len(uid2aid)  # the number of units

        # The maximum number of steps per episode;
        # the length of an episode should never exceed this value
        self.max_episode_steps = kwargs['max_episode_steps']
        self.reward_config = reward_config
        self.use_action_mask = True  # if action mask is used
        self.use_life_mask = True  # if life mask is used

        self._unity_disc_actions = 2
        self._unity_cont_actions = 2
        self._action_dim = 5
        self._obs_dim = 61
        self._global_state_dim = 61
        self.action_space = [gym.spaces.Discrete(self._action_dim) for _ in range(self.n_agents)]
        self.blue_agent = Opponent(self.n_envs)

        # We expect <obs> in self.reset and self.step to return a list of dicts,
        # each associated to an agent. self.obs_shape and self.obs_dtype specify
        # the corresponding shape and dtype.
        # We do not consider the environment and player dimensions here!
        self.obs_shape = [dict(
            obs=self._get_obs_shape(aid),
            global_state=self._get_global_state_shape(aid),
        ) for aid in range(self.n_agents)]
        self.obs_dtype = [dict(
            obs=np.float32,
            global_state=np.float32,
        ) for _ in range(self.n_agents)]
        if self.use_life_mask:
            for aid in range(self.n_agents):
                # 1 if the unit is alive, otherwise 0
                self.obs_shape[aid]['life_mask'] = ()
                self.obs_dtype[aid]['life_mask'] = np.float32
        if self.use_action_mask:
            for aid in range(self.n_agents):
                # 1 if the action is valid, otherwise 0
                self.obs_shape[aid]['action_mask'] = (self.action_space[aid].n,)
                self.obs_dtype[aid]['action_mask'] = bool

        # The following stats should be updated in self.step and be reset in self.reset
        # The episodic score we use to evaluate agent's performance. It excludes shaped rewards
        self._score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        # The accumulated episodic rewards we give to the agent. It includes shaped rewards
        self._dense_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        # The length of the episode
        self._epslen = np.zeros(self.n_envs, np.int32)
        self.alive_units = np.ones((self.n_envs, len(self._red_names + self._blue_names)), np.int32)
        self.detect_units = np.zeros((self.n_envs, len(self._blue_names)), np.bool)

    def random_action(self):
        actions = []
        for aid, uids in enumerate(self.aid2uids):
            a = np.array([[self.action_space[aid].sample() for _ in uids] for _ in range(self.n_envs)], np.int32)
            actions.append(a)
        return actions

    def reset(self):
        self._score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._dense_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._epslen = np.zeros(self.n_envs, np.int32)

        self.alive_units = np.ones((self.n_envs, len(self._red_names + self._blue_names)), np.int32)
        self.detect_units = np.zeros((self.n_envs, len(self._blue_names)), np.bool)

        decision_steps, terminal_steps = self.env.reset()
        # decision_steps, terminal_steps = {}, {}

        return self._get_obs(decision_steps)

    def step(self, actions):
        # TODO: auto-reset when done is True or when max_episode_steps meets
        # NOTE: this should be done environment-wise.
        # It will not be an easy task; please take extra care to make it right!
        self._set_blue_action(self._blue_names, self.blue_agent.actions)

        self._set_action(self._red_names, actions)
        decision_steps, terminal_steps = self.env.step()

        self._epslen += 1

        if len(terminal_steps[self._blue_names[0]]) != 0:
            decision_steps = terminal_steps

        #     return
        # self._set_copy_need_reset(done, self._epslen)
        # TODO 结束的某个环境的reset
        # if len(terminal_steps[self._blue_names[0]]) != 0:
        #     print()
        #     #decision_steps, terminal_steps = self.env.step()
        #     decision_steps, terminal_steps = self.env.reset()
        #     print()
        #     #return self._get_obs(decision_steps, terminal_steps)

        agent_obs = self._get_obs(decision_steps)

        # reward = np.random.normal(size=self.n_envs)
        # done = np.random.randint(0, 2, self.n_envs, dtype=bool)
        reward = self._get_reward()
        done = self._get_done(terminal_steps)

        discount = 1 - done  # we return discount instead of done

        assert reward.shape == (self.n_envs,), reward.shape
        assert discount.shape == (self.n_envs,), discount.shape
        # obtain ndarrays of shape (n_envs, n_units)
        rewards = np.tile(reward.reshape(-1, 1), (1, self.n_units))
        discounts = np.tile(discount.reshape(-1, 1), (1, self.n_units))

        # TODO: these stats should be updated accordingly when auto resetting
        self._dense_score += rewards
        self._score += np.where(discounts, 0, rewards > 0)  # an example for competitive games

        for i in range(self.n_envs):
            if self._epslen >= self.max_episode_steps:
                done[i] = True

        reset = done
        assert reset.shape == (self.n_envs,), reset.shape
        resets = np.tile(reset.reshape(-1, 1), (1, self.n_units))

        self._info = [dict(
            score=self._score[i],
            dense_score=self._dense_score[i],
            epslen=self._epslen[i],
            game_over=discount[i] == 0
        ) for i in range(self.n_envs)]

        # group stats of units controlled by the same agent
        agent_reward = [rewards[:, uids] for uids in self.aid2uids]
        agent_discount = [discounts[:, uids] for uids in self.aid2uids]
        agent_reset = [resets[:, uids] for uids in self.aid2uids]

        for i in range(self.n_envs):
            if done[i]:
                #self.env.reset_envs_with_ids([i])
                self.reset()
                self._dense_score[i] = 0
                self._score[i] = 0
                self._epslen[i] = 0

                self.alive_units[i] = np.ones(len(self._red_names + self._blue_names), np.int32)
                self.detect_units[i] = np.zeros(len(self._blue_names), np.bool)

        if len(terminal_steps[self._blue_names[0]]) != 0:
            assert np.any(reward < 0), reward
            assert np.any(resets), resets

        # we return agent-wise data
        return agent_obs, agent_reward, agent_discount, agent_reset

    def info(self):
        return self._info

    def close(self):
        # close the environment
        pass

    def _get_obs_shape(self, aid):
        return (self._obs_dim,)
        # return ((aid+1) * 2,)

    def _get_global_state_shape(self, aid):
        return (self._global_state_dim,)
        # return ((aid+1) * 1,)

    def _get_obs(self, decision_step):
        obs = [{} for _ in range(self.n_agents)]
        # for aid in range(self.n_agents):
        #     for k in self.obs_shape[aid].keys():
        #         obs[aid][k] = np.zeros(
        #             (self.n_envs, len(self.aid2uids[aid]), *self.obs_shape[aid][k]),
        #             dtype=self.obs_dtype[aid][k]
        #         )
        #         assert obs[aid][k].shape == (self.n_envs, len(self.aid2uids[aid]), *self.obs_shape[aid][k]), \
        #             (obs[aid][k].shape, (self.n_envs, len(self.aid2uids[aid]), *self.obs_shape[aid][k]))
        red_info = []
        for red in self._red_names:
            red_info.append(decision_step[red].obs)

        blue_info = []
        for blue in self._blue_names:
            blue_info.append(decision_step[blue].obs)

        all_obs = self._get_all_units_obs(red_info, blue_info)
        for aid in range(self.n_agents):
            for k in self.obs_shape[aid].keys():
                obs[aid][k] = np.zeros(
                    (self.n_envs, len(self.aid2uids[aid]), *self.obs_shape[aid][k]),
                    dtype=self.obs_dtype[aid][k]
                )
                for n in range(self.n_envs):
                    for id, uid in enumerate(self.aid2uids[aid]):
                        obs[aid][k][n][id] = all_obs[n][k][uid]

        return obs

    def _get_all_units_obs(self, red_info, blue_info):
        blue_ray_info = np.array([i[0] for i in blue_info]).swapaxes(0, 1)
        blue_state_info = np.array([i[1][:, 0:9] for i in blue_info]).swapaxes(0, 1)
        blue_missile_info = np.array([i[1][:, 9:] for i in blue_info]).swapaxes(0, 1)

        red_main_ray_info = np.array([red_info[0][0]]).swapaxes(0, 1)
        red_ally_ray_info = np.array([i[0] for i in red_info[1:]]).swapaxes(0, 1)
        red_state_info = np.array([i[1][:, 0:9] for i in red_info]).swapaxes(0, 1)
        red_missile_info = np.array([red_info[0][1][:, 9:]]).swapaxes(0, 1)

        all_position = self._get_all_plane_position(np.concatenate((blue_state_info, red_state_info), axis=1))

        self._update_blue_alive(blue_state_info)
        blue_observation = [blue_ray_info, blue_state_info, blue_missile_info, red_state_info, red_missile_info]
        self.blue_agent.choose_action(self.env, blue_observation)

        red_observation = []
        for i in range(self.n_envs):
            observations = {}
            global_state, is_alive, action_mask = [], [], []

            for v, k in enumerate(self._red_names):
                index = red_state_info[i][v][0]
                alive = red_state_info[i][v][1]
                self.alive_units[i][v + 2] = alive if self.alive_units[i][v + 2] != -1 else -1
                position = red_state_info[i][v][2:4]
                vel = red_state_info[i][v][4:6]
                angle = red_state_info[i][v][6]
                direction = red_state_info[i][v][7:9]
                angle_ = [math.sin(angle), math.cos(angle)]
                left_missile = red_missile_info[i][0][1] if v == 0 else 0

                info = deepcopy(all_position[i])
                red_main_position = info[2][3:5]
                info = np.delete(info, v + 2, 0)
                other_role = info[:, 0:3]
                other_alive = info[:, 5:]

                other_position = info[:, 3:5] - position
                other_distance = np.vstack([np.linalg.norm(i) for i in other_position])
                if v == 0:
                    detect_flag = self._get_detect_flag(red_main_ray_info[i][0], v)
                else:
                    detect_flag = self._get_detect_flag(red_ally_ray_info[i][v - 1], v)
                # rewards[i][v] += sum(detect_flag) * self.reward_setting['DETECT_REWARD']
                self.detect_units[i][0] += detect_flag[0]
                self.detect_units[i][1] += detect_flag[1]

                main_missile = self._get_target_missile(blue_missile_info[i], 0, red_main_position)
                me_missile = self._get_target_missile(blue_missile_info[i], v, position)
                blue_missile_left = blue_missile_info[i][:, 1]

                agent_obs = {'position': position, 'vel': vel, 'direction': angle_,
                             'left_missile': left_missile,
                             'other_role': other_role, 'other_alive': other_alive, 'other_position': other_position,
                             'other_distance': other_distance,
                             'detect_flag': detect_flag, 'me_missile': me_missile, 'main_missile': main_missile,
                             'enemy_missile_left': blue_missile_left
                             }

                if v == 0:
                    if red_missile_info[i][0][1] > 0:
                        if sum(red_missile_info[i][0][-2:]) == 1:
                            action_mask.append([1, 1, 1, 1, 0])
                        if sum(red_missile_info[i][0][-2:]) == 2:
                            action_mask.append([1, 1, 1, 1, 1])
                        if sum(red_missile_info[i][0][-2:]) == 0:
                            action_mask.append([1, 1, 1, 0, 0])
                    else:
                        action_mask.append([1, 1, 1, 0, 0])
                else:
                    action_mask.append([1, 1, 1, 0, 0])

                ret = []
                for obs_k, obs_v in agent_obs.items():
                    ret.extend(np.array(obs_v).flatten())

                global_state.append(np.array(ret))
                is_alive.append(alive)

            observations['obs'] = np.array(global_state)
            observations['global_state'] = np.array(global_state)
            observations['life_mask'] = np.array(is_alive)
            observations['action_mask'] = np.array(action_mask)

            assert action_mask != [1, 1, 1, 0, 1]
            # try:
            #     assert is_alive[0] != 0, (is_alive)
            # except AssertionError:
            #     print()
            #     print()
            red_observation.append(observations)

        return red_observation

    def _update_blue_alive(self, blue_state_info):
        for i in range(self.n_envs):
            for id, blue in enumerate(self._blue_names):
                if blue_state_info[i][id][1] == 0 and self.alive_units[i][id] == 1:
                    self.alive_units[i][id] = 0

    def _get_all_plane_position(self, all_states):
        n_ret = [0] * self.n_envs
        for i in range(self.n_envs):
            ret = []
            for v, k in enumerate(self._blue_names + self._red_names):
                info = all_states[i][v]
                role = self._get_plane_role(str(k))
                if role == 0:
                    role_array = np.array([1, 0, 0])
                if role == 1:
                    role_array = np.array([0, 1, 0])
                if role == 2:
                    role_array = np.array([0, 0, 1])

                position = info[2:4]
                alive = [1, 0] if info[1] == 1 else [0, 1]

                ret = np.append(np.append(role_array, position), alive) if len(ret) == 0 else \
                    np.vstack((ret, np.append(np.append(role_array, position), alive)))
            n_ret[i] = ret
        return np.array(n_ret)

    def _get_plane_role(self, name):
        if name.startswith('red_main'):
            return 0
        if name.startswith('red_sup'):
            return 1
        if name.startswith('blue'):
            return 2

    def _get_detect_flag(self, ray_info, agent_id):
        if agent_id == 0:
            tag_num = 4
            ray_length = self._red_main_ray_dim
            enemy_index = 2
        else:
            tag_num = 3
            ray_length = self._red_ally_ray_dim
            enemy_index = 1
        detect_flag = []
        for i in range(enemy_index, ray_length, tag_num + 2):
            if ray_info[i] == 1:
                detect_flag.append(1)
        if len(detect_flag) == 0:
            detect_flag = [0, 0]
        if len(detect_flag) == 1:
            detect_flag.append(0)
        if len(detect_flag) > 2:
            detect_flag = [1, 1]
        # detect_flag = detect_flag.extend([0]*(2-len(detect_flag)))
        return np.array(detect_flag)

    def _get_enemy_missile(self, blue_data):
        missile_data_index = self._blue_ray_dim + 10
        missile_data = []
        missile_left = []
        for i in blue_data:
            missile_left.append(i[missile_data_index - 1])
            for j in range(missile_data_index, i.size, 9):
                if i[missile_data_index + 8] == 1:
                    missile_data.append([i[missile_data_index + 7], i[missile_data_index], i[missile_data_index + 1]])

        return np.array(missile_data), np.array(missile_left)

    def _get_target_missile(self, enemy_missile, target, position):
        distance = 0
        for id, missile in enumerate(enemy_missile):
            for j in range(2, len(missile) - 5 - 9 + 1, 9):
                if missile[j + 7] == target and missile[j + 8] == 1:
                    m_dis = np.linalg.norm(missile[j:j + 2] - position)
                    distance = min(distance, m_dis) if distance != 0 else m_dis
        return distance

    def _set_blue_action(self, names, actions):
        for uid, name in enumerate(names):
            action_tuple = self.env.get_action_tuple()
            action_tuple.add_discrete(actions[name][:, 2:4])
            action_tuple.add_continuous(actions[name][:, 0:2])

            self.env.set_actions(name, action_tuple)


    def _set_action(self, names: List[str], actions: np.ndarray):
        """ Set actions for names
        actions is produced by an Agent instance, expected to be of shape [N_ENVS, N_UNITS]
        """

        def to_unity_action(action):
            if action == 0:
                action = [0, 1], [0, 0]
            elif action == 1:
                action = [-1, 1], [0, 0]
            elif action == 2:
                action = [1, 1], [0, 0]
            elif action == 3:
                action = [0, 1], [1, 0]
            elif action == 4:
                action = [0, 1], [0, 1]
            else:
                raise ValueError(f'Invalid action: {action}')
            return action

        actions = numpy.concatenate(actions, axis=1)

        assert len(names) == actions.shape[1], (names, actions.shape)
        for uid, name in enumerate(names):
            dist_action, cont_action = zip(*[to_unity_action(a) for a in actions[:, uid]])
            dist_action = np.array(dist_action, np.int32)
            cont_action = np.array(cont_action, np.float32)
            action_tuple = self.env.get_action_tuple()
            action_tuple.add_discrete(dist_action)
            action_tuple.add_continuous(cont_action)
            self.env.set_actions(name, action_tuple)

    def _get_reward(self):
        reward = np.zeros(self.n_envs)
        for i in range(self.n_envs):
            if self.alive_units[i][len(self._blue_names)] == 0:
                reward[i] += self.reward_config['main_dead_reward']
                self.alive_units[i][len(self._blue_names)] == -1
            for j in range(len(self._blue_names)):
                if self.alive_units[i][j] == 0:
                    reward[i] += self.reward_config['blue_dead_reward']
                    self.alive_units[i][j] = -1
                reward[i] += self.reward_config['detect_reward'] if self.detect_units[i][j] != 0 else 0

        return reward

    def _get_done(self, terminal_steps):
        dones = {}
        for bn in terminal_steps.keys():
            dones[bn] = np.zeros(self.n_envs, bool)
            dones[bn][terminal_steps[bn].agent_id % self.n_envs] = True
        done = next(iter(dones.values()))
        for bn, d in dones.items():
            assert (d == done).all(), (bn, d, done)

        return done

    def _set_copy_need_reset(self, done, epslen):
        reset_id = []
        for i in range(self.n_envs):
            if done[i] is True or epslen[i] >= self.max_episode_steps:
                reset_id.append(i)
        self.env.reset_envs_with_ids(reset_id)


""" Test Code """
if __name__ == '__main__':
    config = dict(
        env_name='dummy',
        uid2aid=[0, 1, 1, 1, 1],
        max_episode_steps=400,
        n_envs=1,
        unity_config={
            #'worker_id': 0,
            #'file_name':'E:\FlightCombat\FightSimulator\FightSimulator\Packages\\test2\T2.exe'
        },
        reward_config={
            'detect_reward': 0.2, 'main_dead_reward': -10, 'blue_dead_reward': 10
        }
    )


    def print_dict(d, prefix=''):
        for k, v in d.items():
            if isinstance(v, dict):
                print(f'{prefix} {k}')
                print_dict(v, prefix + '\t')
            elif isinstance(v, tuple):
                # namedtuple is assumed
                print(f'{prefix} {k}')
                print_dict(v._asdict(), prefix + '\t')
            else:
                print(f'{prefix} {k}: {v}')


    def print_dict_tensors(d, prefix=''):
        for k, v in d.items():
            if isinstance(v, dict):
                print(f'{prefix} {k}')
                print_dict_tensors(v, prefix + '\t')
            elif isinstance(v, tuple):
                # namedtuple is assumed
                print(f'{prefix} {k}')
                print_dict_tensors(v._asdict(), prefix + '\t')
            else:
                print(f'{prefix} {k}: {v.shape} {v.dtype}')

    n_unity_file = 1
    n_unity_env = []
    for n in range(n_unity_file):
        n_unity_env.append(UnityEnv(**config))
        #config['unity_config']['worker_id'] = config['unity_config']['worker_id'] + 1

    #assert False
    env = n_unity_env[0]
    observations = env.reset()
    print('reset observations')
    for i, o in enumerate(observations):
        print_dict_tensors(o, f'\tagent{i}')

    for k in range(1, 5000):
        # env.env.step()
        actions = env.random_action()
        print(f'Step {k}, random actions', actions)
        observations, rewards, dones, reset = env.step(actions)
        print(f'Step {k}, observations')
        for i, o in enumerate(observations):
            print_dict_tensors(o, f'\tagent{i}')
        print(f'Step {k}, rewards', rewards)
        print(f'Step {k}, dones', dones)
        print(f'Step {k}, reset', reset)
        info = env.info()
        print(f'Step {k}, info')
        for aid, i in enumerate(info):
            print_dict(i, f'\tenv{aid}')
