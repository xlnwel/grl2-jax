import functools
from typing import List

import numpy
import numpy as np
import gym

from interface import UnityInterface
from opponent import Opponent

"""  Naming Conventions:
An Agent is the one that makes decisions, it's typically a RL agent
A Unit is a controllable unit in the environment.
Each agent may control a number of units.
All ids start from 0 and increases sequentially
"""

WALL = 50
MAX_LEN = 50


def compute_aid2uids(uid2aid):
    """ Compute aid2uids from uid2aid """
    aid2uids = []
    for uid, aid in enumerate(uid2aid):
        if aid > len(aid2uids):
            raise ValueError(f'uid2aid({uid2aid}) is not sorted in order')
        if aid == len(aid2uids):
            aid2uids.append((uid,))
        else:
            aid2uids[aid] += (uid,)
    aid2uids = [np.array(uids, np.int32) for uids in aid2uids]

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
                'detect_reward': 0,
                'main_dead_reward': -10,
                'blue_dead_reward': 10,
                'grid_reward': 0.1
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

        # unity_config['n_envs'] = n_envs
        self.env = UnityInterface(**unity_config)

        teams = {}
        for name in self.env.get_behavior_names():
            team = name[name.index('team'):]
            if teams.__contains__(team) is False:
                teams[team] = [name]
            else:
                teams[team].append(name)

        self._red_names = []
        self._blue_names = []

        def team_cmp(t1, t2):
            k1 = str(t1[0])
            k2 = str(t2[0])
            team1 = int(k1[k1.index('=') + 1:])
            team2 = int(k2[k2.index('=') + 1:])
            if team1 < team2:
                return -1
            if team1 == team2:
                return 0
            if team1 > team2:
                return 1

        for k, v in sorted(teams.items(), key=functools.cmp_to_key(team_cmp)):
            self._red_names.append([name for name in sorted(v) if name.startswith('red')])
            self._blue_names.append([name for name in sorted(v) if name.startswith('blue')])

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
        self._action_dim = [5, 5]
        self._obs_dim = [48, 48]
        self._global_state_dim = [48, 48]
        self.action_space = [gym.spaces.Discrete(ad) for ad in self._action_dim]
        self.blue_agent = Opponent(self.n_envs)

        # We expect <obs> in self.reset and self.step to return a list of dicts,
        # each associated to an agent. self.obs_shape and self.obs_dtype specify
        # the corresponding shape and dtype.
        # We do not consider the environment and player dimensions here!
        self.obs_shape = [dict(
            obs=self._get_obs_shape(aid),
            global_state=self._get_global_state_shape(aid),
            prev_reward=(),
            prev_action=(self._action_dim[aid],),
        ) for aid in range(self.n_agents)]
        self.obs_dtype = [dict(
            obs=np.float32,
            global_state=np.float32,
            prev_reward=np.float32,
            prev_action=np.float32,
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

        # 不同的score，分别对应击毁蓝方、自己死亡、检测到蓝方、蛋爆炸距离的reward
        self._dense_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._win_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._lose_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._detect_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._missile_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)

        # The length of the episode
        self._epslen = np.zeros(self.n_envs, np.int32)

        # 红蓝双方存活状况
        self.alive_units = np.ones((self.n_envs, self.n_units + 2), np.int32)

        # 红方检测到蓝方的情况
        self.detect_units = np.zeros((self.n_envs, self.n_units, len(self._blue_names[0])), np.bool)
        # 双方检测到的累计步数
        self.detect_steps = np.zeros((self.n_envs, self.n_units + 2), np.int32)
        # 存活步数
        self.alive_steps = np.zeros((self.n_envs, self.n_units + 2), np.int32)

        self.blue_actions = {}

        #self.flying_missile = np.ones((self.n_envs, 2), np.int32)
        #self.grid_map = np.zeros((self.n_envs, 10, 10), np.int32)
        # 红方是否被锁定，-1为未被锁定，非负表示锁定的蛋，-2表示锁定的已爆（计算爆炸reward，然后重新赋值为-1
        # 锁定的蛋的距离及坐标
        self.missile_locked = -1 * np.ones((self.n_envs, self.n_units))
        self.missile_dis = MAX_LEN * np.ones((self.n_envs, self.n_units))
        self.missile_position = np.zeros((self.n_envs, self.n_units, 2))

        self.enemy_position = np.zeros((self.n_envs, 2, 4))

        self.pre_reward = np.zeros((self.n_envs, self.n_units), np.float32)
        self.prev_action = None
        self._consecutive_action = np.zeros((self.n_envs, self.n_units), bool)
        self.win_games = 0
        self.lose_games = 0
        self.all_games = 0
        self.no_missile = 0
        self.max_steps_end = 0
        self.total_game = 0

    def random_action(self):
        actions = []
        for aid, uids in enumerate(self.aid2uids):
            a = np.array([[self.action_space[aid].sample() for _ in uids] for _ in range(self.n_envs)], np.int32)
            actions.append(a)
        return actions

    def reset(self):
        self._score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._dense_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._win_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._lose_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._detect_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._missile_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)

        self._epslen = np.zeros(self.n_envs, np.int32)

        self.alive_units = np.ones((self.n_envs, len(self._red_names[0] + self._blue_names[0])), np.int32)
        self.detect_units = np.zeros((self.n_envs, self.n_units, len(self._blue_names[0])), np.bool)
        self.flying_missile = np.ones((self.n_envs, 2), np.int32)
        self.grid_map = np.zeros((self.n_envs, 10, 10), np.int32)
        self.missile_locked = -1 * np.ones((self.n_envs, self.n_units))
        self.missile_dis = MAX_LEN * np.ones((self.n_envs, self.n_units))
        self.missile_position = np.zeros((self.n_envs, self.n_units, 2))
        self.enemy_position = np.zeros((self.n_envs, 2, 4))
        self.pre_reward = np.zeros((self.n_envs, self.n_units), np.float32)
        self._consecutive_action = np.zeros((self.n_envs, self.n_units), bool)
        self.prev_action = None

        decision_steps, terminal_steps = self.env.reset()

        return self._get_obs(decision_steps)

    def step(self, actions):
        self._set_blue_action(self.blue_actions)
        self._set_action(self._red_names, actions)

        decision_steps, terminal_steps = self.env.step()

        self._epslen += 1

        # TODO: Add previous actions to the observations
        agent_obs = self._get_obs(decision_steps, self.pre_reward)

        done, reward, win_r, lose_r, detect_r, missile_r = self._get_done_and_reward()

        alive_blue = np.zeros(self.n_envs)
        alive_main = np.zeros(self.n_envs)
        alive_ally = np.zeros(self.n_envs)
        # 更新存活情况，用于info记录
        for i in range(self.n_envs):
            if done[i]:
                alive_main[i] = decision_steps[self._red_names[i][0]].obs[1][0][1]
                alive_ally[i] = 0
                alive_blue[i] = 0

                for name in self._red_names[i][1:]:
                    if decision_steps[name].obs[1][0][1] == 1:
                        alive_ally[i] += 1
                for name in self._blue_names[i]:
                    if decision_steps[name].obs[1][0][1] == 1:
                        alive_blue[i] += 1
            else:
                for j in self.alive_units[i][0:2]:
                    if j == 1:
                        alive_blue[i] += j
                alive_main[i] = 1 if self.alive_units[i][2] == 1 else 0
                for j in self.alive_units[i][3:7]:
                    if j == 1:
                        alive_ally[i] += j

        for i in range(self.n_envs):
            if self._epslen[i] > self.max_episode_steps:
                done[i] = True
                #print('max steps')
                self.max_steps_end += 1

        discount = 1 - done  # we return discount instead of done

        assert reward.shape == (self.n_envs, self.n_units), reward.shape
        assert discount.shape == (self.n_envs,), discount.shape
        # obtain ndarrays of shape (n_envs, n_units)
        # rewards = np.tile(reward.reshape(-1, 1), (1, self.n_units))
        rewards = reward
        discounts = np.tile(discount.reshape(-1, 1), (1, self.n_units))

        self._dense_score += rewards
        self._win_score += win_r
        self._lose_score += lose_r
        self._detect_score += detect_r
        self._missile_score += missile_r

        # self._score += np.where(discounts, 0, rewards > 0)  # an example for competitive games
        reset = done
        assert reset.shape == (self.n_envs,), reset.shape
        resets = np.tile(reset.reshape(-1, 1), (1, self.n_units))

        if self.prev_action is not None:
            for tt in range(self.n_envs):
                for aid in range(self.n_agents):
                    for id, uid in enumerate(self.aid2uids[aid]):
                        pa = self.prev_action[aid][tt][uid - aid]
                        a = actions[aid][tt][uid - aid]
                        self._consecutive_action[tt][uid] = (pa == a)
        self.prev_action = actions

        self._info = [dict(
            win_game=self.win_games,
            lose_game=self.lose_games,
            no_missile_end_game=self.no_missile,
            max_steps_end_game=self.max_steps_end,
            total_game=self.total_game,
            score=self._score[i].copy(),
            dense_score=self._dense_score[i].copy(),
            win_score=self._win_score[i].copy(),
            lose_score=self._lose_score[i].copy(),
            detect_score=self._detect_score[i].copy(),
            missile_score=self._missile_score[i].copy(),
            epslen=self._epslen[i],
            game_over=discount[i] == 0,
            consecutive_action=self._consecutive_action[i].copy(),
            left_missile_red=self.missile_end[i][2],
            left_missile_blue0=self.missile_end[i][0],
            left_missile_blue1=self.missile_end[i][1],
            alive_blue=alive_blue[i],
            alive_main=alive_main[i],
            alive_ally=alive_ally[i],
            main_detect_steps=self.detect_steps[i][2],
            ally0_detect_steps=self.detect_steps[i][3],
            ally1_detect_steps=self.detect_steps[i][4],
            ally2_detect_steps=self.detect_steps[i][5],
            ally3_detect_steps=self.detect_steps[i][6],
            main_alive_steps=self.alive_steps[i][2],
            ally_alive_steps=self.alive_steps[i][3:].mean(),
            blue_detect_main=self.detect_steps[i][0],
            blue_detect_ally=self.detect_steps[i][1],
            blue0_alive_steps=self.alive_steps[i][0],
            blue1_alive_steps=self.alive_steps[i][1]
        ) for i in range(self.n_envs)]
        for i in range(self.n_envs):
            assert self.red_missile_left[i] <= 4
            assert self.blue_missle_left[i][0] <= 3
            assert self.blue_missle_left[i][1] <= 3

        # group stats of units controlled by the same agent
        agent_reward = [rewards[:, uids] for uids in self.aid2uids]
        agent_discount = [discounts[:, uids] for uids in self.aid2uids]
        agent_reset = [resets[:, uids] for uids in self.aid2uids]

        reset_env = []
        for i in range(self.n_envs):
            if done[i]:
                self.total_game += 1
                reset_env.append(i + 1)
                self._dense_score[i] = np.zeros(self.n_units, np.float32)
                self._win_score[i] = np.zeros(self.n_units, np.float32)
                self._lose_score[i] = np.zeros(self.n_units, np.float32)
                self._detect_score[i] = np.zeros(self.n_units, np.float32)
                self._missile_score[i] = np.zeros(self.n_units, np.float32)

                self._score[i] = np.zeros(self.n_units, np.float32)
                self._epslen[i] = 0

                self.alive_units[i] = np.ones(len(self._red_names[i] + self._blue_names[i]), np.int32)
                self.detect_units[i] = np.zeros((self.n_units, len(self._blue_names[i])), np.bool)
                self.detect_steps[i] = np.zeros(self.n_units + 2)
                self.alive_steps[i] = np.zeros(self.n_units + 2)
                self.flying_missile[i] = np.ones(2, np.int32)
                self.grid_map[i] = np.zeros((10, 10), np.int32)
                self.missile_locked[i] = -1 * np.ones(self.n_units)
                self.missile_dis[i] = MAX_LEN * np.ones(self.n_units)
                self.missile_position[i] = np.zeros((self.n_units, 2))
                self.enemy_position[i] = np.zeros((2, 4))
                self.prev_action = None

        self.env.reset_envs_with_ids(reset_env) if len(reset_env) != 0 else None

        return agent_obs, agent_reward, agent_discount, agent_reset

    def info(self):
        return self._info

    def close(self):
        # close the environment
        pass

    def _get_obs_shape(self, aid):
        return (self._obs_dim[aid],)

    def _get_global_state_shape(self, aid):
        return (self._global_state_dim[aid],)

    def _get_obs(self, decision_step, reward=None, action=None):
        obs = [{} for _ in range(self.n_agents)]

        red_info = []

        # 根据team分别获取数据
        for team in self._red_names:
            t_red = []
            for red in team:
                t_red.append(decision_step[red].obs)
            red_info.append(t_red)

        blue_info = []
        for team in self._blue_names:
            t_blue = []
            for blue in team:
                t_blue.append(decision_step[blue].obs)
            blue_info.append(t_blue)

        all_obs = self._get_all_units_obs(red_info, blue_info, reward, action)

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

    def _get_all_units_obs(self, red_info, blue_info, reward, action):
        self.blue_actions = {}
        red_observation = []
        self.blue_missle_left = np.zeros((self.n_envs, 2), np.int32)
        self.red_missile_left = np.zeros(self.n_envs, np.int32)

        # missile_end记录蛋是否已爆
        self.missile_end = np.zeros((self.n_envs, 3))
        self.detect_units = np.zeros((self.n_envs, self.n_units, len(self._blue_names[0])), np.bool)

        for team in range(self.n_envs):
            # 获取蓝方射线信息、状态信息、导弹信息
            blue_ray_info = np.array([i[0] for i in blue_info[team]]).squeeze()
            blue_state_info = np.array([i[1][:, 0:9] for i in blue_info[team]]).squeeze()
            blue_missile_info = np.array([i[1][:, 9:] for i in blue_info[team]]).squeeze()

            # 获取红方主僚机的射线信息、状态信息、导弹信息
            red_main_ray_info = np.array([red_info[team][0][0]]).squeeze()
            red_ally_ray_info = np.array([i[0] for i in red_info[team][1:]]).squeeze()
            red_state_info = np.array([i[1][:, 0:9] for i in red_info[team]]).squeeze()
            red_missile_info = np.array([red_info[team][0][1][:, 9:]]).squeeze()

            # 根据状态信息获取获取各个飞机相互之间的相对坐标、距离、角色等
            all_position = self._get_all_plane_position(team, np.concatenate((blue_state_info, red_state_info), axis=0))

            # 更新蓝方的存活数据
            self._update_blue_alive(team, blue_state_info)

            blue_observation = [blue_ray_info, blue_state_info, blue_missile_info,
                                red_state_info, red_missile_info]
            # 获取蓝方探测到红方的数据、蓝方的决策动作
            detect_red, b_a = self.blue_agent.choose_action(team, blue_observation)
            self.detect_steps[team][0] += 1 if detect_red[0] != 0 else 0
            self.detect_steps[team][1] += 1 if detect_red[1] != 0 else 0

            self.blue_actions[self._blue_names[team][0]] = b_a[0]
            self.blue_actions[self._blue_names[team][1]] = b_a[1]
            assert len(self.blue_actions) == 2 * (team + 1), self.blue_actions

            self.blue_missle_left[team] = blue_missile_info[:, 1]

            # 更新蓝方蛋的生命周期
            self._update_missile_life(blue_missile_info, team)

            observations = {}
            global_state, is_alive, action_mask = [], [], []

            # 遍历红方飞机，计算observation
            for v, k in enumerate(self._red_names[team]):
                index = red_state_info[v][0]
                alive = red_state_info[v][1]
                self.alive_units[team][v + 2] = alive if self.alive_units[team][v + 2] != -1 else -1
                position = red_state_info[v][2:4]
                grid_x = int((50 - position[0]) // 50)
                grid_z = int((50 - position[1]) // 50)
                if self.grid_map[team][grid_x][grid_z] == 0:
                    self.grid_map[team][grid_x][grid_z] = 1
                vel = red_state_info[v][4:6]
                angle = red_state_info[v][6]
                direction = red_state_info[v][7:9]
                left_missile = red_missile_info[1] if v == 0 else 0
                if v == 0:
                    self.red_missile_left[team] = left_missile

                info = all_position.copy()
                red_main_position = info[2][3:5]
                info = np.delete(info, v + 2, 0) # 删除自己的信息
                other_role = info[:, 0:3]
                other_alive = sum(info[0:2, 5])
                info[0:2, 3:5] = self.enemy_position[team, :, 0:2] # 用env维护的蓝方位置信息替换真实位置信息

                other_position = info[:, 3:5] - position
                other_distance = np.vstack([np.linalg.norm(i) for i in other_position])
                # 获取是否探测到蓝方
                if v == 0:
                    detect_flag = self._get_detect_flag(red_main_ray_info, v)
                else:
                    detect_flag = self._get_detect_flag(red_ally_ray_info[v - 1], v)
                # rewards[i][v] += sum(detect_flag) * self.reward_setting['DETECT_REWARD']
                self.detect_units[team][v][0] += detect_flag[0]
                self.detect_units[team][v][1] += detect_flag[1]
                self.detect_steps[team][v + 2] += 1 if sum(detect_flag) > 0 else 0
                self.alive_steps[team][v + 2] += 1 if alive == 1 else 0

                # for bid in range(2):
                #     self.enemy_position[team][bid] = blue_state_info[bid, 2:6]
                # 蓝方位置信息
                for bid in range(2):
                    if detect_flag[bid] == 1:
                        self.enemy_position[team][bid] = blue_state_info[bid, 2:6]
                    else:
                        if self._epslen[team] % 7 == 0:
                            self.enemy_position[team][bid] = blue_state_info[bid, 2:6] + np.random.randn(4)
                        else:
                            self.enemy_position[team][bid][0:2] = self._predict_position(
                                self.enemy_position[team][bid][0:2],
                                self.enemy_position[team][bid][2:4],
                                self._epslen[team] % 7)
                # 锁定主机和自己的最近的蛋
                main_missile = self._get_target_missile(blue_missile_info, team, 0, red_main_position)
                me_missile = self._get_target_missile(blue_missile_info, team, v, position)
                # if me_missile != 0:
                #     assert me_missile < MAX_LEN, me_missile
                #     self.missile_dis[team][v] -= me_missile
                blue_missile_left = blue_missile_info[:, 1]
                agent_obs = {'vel': vel, 'direction': direction,
                             'left_missile': left_missile,
                             'other_role': other_role, 'other_alive': other_alive, 'other_position': other_position,
                             'other_distance': other_distance,
                             'detect_flag': detect_flag, 'me_missile': me_missile, 'main_missile': main_missile,
                             'enemy_missile_left': blue_missile_left
                             }

                if v == 0:
                    # if self._epslen[team] < 10:
                    #     if red_missile_info[-2] == 1 or red_missile_info[-1] ==1:
                    #         print(self._epslen[team])
                    #         print(red_missile_info[-2:])
                    flying_missile = [1, 1]

                    # 遍历主机的导弹信息，用于控制连续发蛋和已爆的蛋数
                    for j in range(9, 37, 9):
                        #
                        #     if self.flying_missile[team][int(red_missile_info[j])] == 1:
                        if red_missile_info[j + 1] == 1:
                            flying_missile[int(red_missile_info[j])] = 0
                            #self.flying_missile[team][int(red_missile_info[j])] = 0
                        if red_missile_info[j + 1] == 2:
                            #self.flying_missile[team] = 1
                            self.missile_end[team][2] += 1

                    a = [1, 1, 1, 1, 1]
                    # 根据可打击目标和飞行中的蛋计算action mask
                    a[-2:] = red_missile_info[-2:]
                    a[-2] = int(red_missile_info[-2] and flying_missile[-2])
                    a[-1] = int(red_missile_info[-1] and flying_missile[-1])

                    action_mask.append(a)
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
            observations['prev_reward'] = np.zeros(self.n_units, np.float32) if reward is None else reward[team]
            observations['prev_action'] = np.zeros((self.n_units, self._action_dim[team]),
                                                np.float32) if action is None else action[team]

            red_observation.append(observations)

        return red_observation

    def _update_missile_life(self, blue_missile_info, team):
        for id, missile in enumerate(blue_missile_info):
            for j in range(2, len(missile) - 5 - 9 + 1, 9):
                if missile[j + 8] == 2:
                    self.missile_end[team][id] += 1
                    for rid, red in enumerate(self.missile_locked[team]):
                        if red == j // 9:
                            self.missile_locked[team][rid] = -2

    def _update_blue_alive(self, team, blue_state_info):
        for id, blue in enumerate(self._blue_names[team]):
            if blue_state_info[id][1] == 1:
                self.alive_steps[team][id] += 1
            if blue_state_info[id][1] == 0 and self.alive_units[team][id] == 1:
                self.alive_units[team][id] = 0

    def _get_all_plane_position(self, team, all_states):
        ret = []
        for v, k in enumerate(self._blue_names[team] + self._red_names[team]):
            info = all_states[v]
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
        return ret

    def _get_plane_role(self, name):
        if name.startswith('red_main'):
            return 0
        if name.startswith('red_sup'):
            return 1
        if name.startswith('blue'):
            return 2

    def _predict_position(self, last_pos, vel, t):
        return last_pos + 0.1 * vel * t

    def _get_detect_flag(self, ray_info, agent_id):
        if agent_id == 0:
            tag_num = 4
            ray_length = self._red_main_ray_dim
            enemy_index = 1
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
        # detect_flag = detect_flag.extend([0]*(2-len(detect_flag))

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

    def _get_target_missile(self, enemy_missile, team, target, position):
        distance = 0
        for id, missile in enumerate(enemy_missile):
            for j in range(2, len(missile) - 5 - 9 + 1, 9):
                if missile[j + 7] == target and missile[j + 8] == 1:
                    if self.missile_locked[team][target] == -1:
                        self.missile_locked[team][target] = j // 9
                    m_dis = np.linalg.norm(missile[j:j + 2] - position)
                    if distance == 0 or m_dis < distance:
                        distance = m_dis
                        self.missile_position[team][target] = missile[j:j + 2] - position

        return distance

    def _set_blue_action(self, actions):
        for name, action in actions.items():
            action_tuple = self.env.get_action_tuple()
            action_tuple.add_discrete(np.reshape(action[2:4], (1, 2)))
            action_tuple.add_continuous(np.reshape(action[0:2], (1, 2)))

            self.env.set_actions(name, action_tuple)

    def _set_action(self, names: List[str], actions: np.ndarray):
        """ Set actions for names
        actions is produced by an Agent instance, expected to be of shape [N_ENVS, N_UNITS]
        """

        def to_unity_action(name, action):
            if name.startswith('red_main'):
                if action == 0:
                    action = np.array([1, 0], np.float32).reshape(1, 2), np.array([0, 0], np.int32).reshape(1, 2)
                elif action == 1:
                    action = np.array([-0.2, -1], np.float32).reshape(1, 2), np.array([0, 0], np.int32).reshape(1, 2)
                elif action == 2:
                    action = np.array([-0.2, 1], np.float32).reshape(1, 2), np.array([0, 0], np.int32).reshape(1, 2)
                elif action == 3:
                    action = np.array([1, 0], np.float32).reshape(1, 2), np.array([0, 1], np.int32).reshape(1, 2)
                elif action == 4:
                    action = np.array([1, 0], np.float32).reshape(1, 2), np.array([1, 1], np.int32).reshape(1, 2)

            else:
                if action == 0:
                    action = np.array([[1, 0]], np.float32), np.zeros((1, 0), np.int32)
                elif action == 1:
                    action = np.array([[1, -1]], np.float32), np.zeros((1, 0), np.int32)
                elif action == 2:
                    action = np.array([[1, 1]], np.float32), np.zeros((1, 0), np.int32)
                elif action == 3:
                    action = np.array([[1, 0]], np.float32), np.zeros((1, 0), np.int32)
                elif action == 4:
                    action = np.array([[1, 0]], np.float32), np.zeros((1, 0), np.int32)

            return action

        actions = numpy.concatenate(actions, axis=1)

        assert len(names) == actions.shape[0], (names, actions.shape)
        for tid, name in enumerate(names):
            for uid, n in enumerate(name):
                cont_action, dist_action = to_unity_action(n, actions[tid][uid])
                action_tuple = self.env.get_action_tuple()
                action_tuple.add_discrete(dist_action)
                action_tuple.add_continuous(cont_action)
                self.env.set_actions(n, action_tuple)

    def _get_done_and_reward(self):
        """  获取游戏逻辑上done和reward
                """
        done = np.array([False] * self.n_envs)
        reward = np.zeros((self.n_envs, self.n_units), np.float32)
        win_r = np.zeros((self.n_envs, self.n_units), np.float32)
        lose_r = np.zeros((self.n_envs, self.n_units), np.float32)
        detect_r = np.zeros((self.n_envs, self.n_units), np.float32)
        missile_r = np.zeros((self.n_envs, self.n_units), np.float32)

        for i in range(self.n_envs):
            # 蓝方主机都死亡、红方主机未死亡且没有蓝方的蛋正在飞行
            if self.alive_units[i][0] != 1 and self.alive_units[i][1] != 1 and self.alive_units[i][2] == 1\
                    (self.missile_end[i][0] + self.blue_missle_left[i][0]) == \
                    (self.missile_end[i][1] + self.blue_missle_left[i][1]) == 3:
                done[i] = True
                reward[i] += self.reward_config['blue_dead_reward']
                win_r[i] += self.reward_config['blue_dead_reward']
                #print('red win')
                self.win_games += 1
                self._score[i] += 1
                if self.alive_units[i][0] == 0:
                    self.alive_steps[i][0] = self._epslen[i]
                if self.alive_units[i][1] == 0:
                    self.alive_steps[i][1] = self._epslen[i]
            # 红方主机死亡
            elif self.alive_units[i][2] != 1:
                done[i] = True
                reward[i] += self.reward_config['main_dead_reward']
                lose_r[i] += self.reward_config['main_dead_reward']
                #print('red lose')
                self.alive_steps[i][2] = self._epslen[i]
                self.lose_games += 1

            # 蓝方主机都活着，但是红蓝双方所有蛋都打完
            if self.alive_units[i][0] == 1 and self.alive_units[i][1] == 1:
                if sum(self.missile_end[i]) == 10:
                    done[i] = True
                    self.no_missile += 1
                    #print('no missile')
            else:
                # 处理蓝方死掉一个飞机，其他活着的飞机都没蛋的情况
                if self.alive_units[i][0] != 1 and self.missile_end[i][2] == 4 \
                        and self.missile_end[i][1] == 3:
                    done[i] = True
                    #print('no missile')
                    self.no_missile += 1

                if self.alive_units[i][1] != 1 and self.missile_end[i][2] == 4 \
                        and self.missile_end[i][0] == 3:
                    done[i] = True
                    #print('no missile')
                    self.no_missile += 1

            if done[i] == 0:
                for j in range(len(self._blue_names[0])):
                    if self.alive_units[i][j] == 0:
                        reward[i] += self.reward_config['blue_dead_reward']
                        win_r[i] += self.reward_config['blue_dead_reward']
                        self.alive_units[i][j] = -1
                        self.alive_steps[i][j] = self._epslen[i]

                for j in range(len(self._red_names[0])):
                    assert sum(self.detect_units[i][j]) <= 2
                    #if self.missile_end[i][2] != 4:
                    reward[i][j] += self.reward_config['detect_reward'] * sum(self.detect_units[i][j])
                    detect_r[i][j] += self.reward_config['detect_reward'] * sum(self.detect_units[i][j])

                    if self.missile_locked[i][j] == -2:
                        bomb_dis = np.linalg.norm(self.missile_position[i][j])
                        if bomb_dis > MAX_LEN:
                            bomb_dis = MAX_LEN - 1
                        # reward[i][j] -= 0.05 * np.log(MAX_LEN - bomb_dis)
                        missile_r[i][j] -= 0.05 * np.log(MAX_LEN - bomb_dis)

                        self.missile_locked[i][j] = -1

                for g_x in range(10):
                    for g_z in range(10):
                        if self.grid_map[i][g_x][g_z] == 1:
                            # reward[i] += self.reward_config['grid_reward']
                            self.grid_map[i][g_x][g_z] = -1

        self.pre_reward = reward.copy()
        return done, reward, win_r, lose_r, detect_r, missile_r


""" Test Code """
if __name__ == '__main__':
    config = dict(
        env_name='dummy',
        uid2aid=[0, 1, 1, 1, 1],
        max_episode_steps=2000,
        n_envs=1,
        unity_config={
            # 'worker_id': 0,
            # 'file_name':'E:\FlightCombat\FightSimulator\FightSimulator\Packages\Demo\T2.exe'
        },
        reward_config={
            'detect_reward': 0.1, 'main_dead_reward': -10, 'blue_dead_reward': 10, 'grid_reward': 0.1
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
        # config['unity_config']['worker_id'] = config['unity_config']['worker_id'] + 1

    # assert False
    env = n_unity_env[0]
    observations = env.reset()
    print('reset observations')
    for i, o in enumerate(observations):
        print_dict_tensors(o, f'\tagent{i}')

    for k in range(1, 50000):
        # env.env.reset_envs_with_ids([2])
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
