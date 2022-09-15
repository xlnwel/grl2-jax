import copy
import functools
from typing import List

import numpy
import numpy as np
import gym
from .interface import UnityInterface
from .opponent import Opponent

"""  Naming Conventions:
An Agent is the one that makes decisions, it's typically a RL agent
A Unit is a controllable unit in the environment.
Each agent may control a number of units.
All ids start from 0 and increases sequentially
"""

MAX_LEN = 50
SHOT_DIS = 40

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
            frame_skip=5,
            is_action_discrete=True, 
            reward_config={
                'detect_reward': 0,
                'main_dead_reward': 0,
                'blue_dead_reward': 1,
                'grid_reward': 0
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
        self.n_agents = 1  # the number of agents
        self.n_units = 1
        self.frame_skip = frame_skip

        # unity_config['n_envs'] = n_envs
        self.env = UnityInterface(**unity_config)
        print('frame_skip')
        print(frame_skip)
        self._red_names, self._blue_names = self._get_names(self.env)

        self.n_red_main = 1
        self.n_blue_main = 2
        self.n_planes = 3

        self._red_main_ray_dim = 42
        self._blue_ray_dim = 42

        # the number of envs running in parallel, which must be the same as the number
        # of environment copies specified when compiing the Unity environment
        self.n_envs = n_envs
        # The same story goes with n_units

        # The maximum number of steps per episode;
        # the length of an episode should never exceed this value
        self.max_episode_steps = kwargs['max_episode_steps']
        self.reward_config = reward_config
        self.use_action_mask = True  # if action mask is used
        self.use_life_mask = True  # if life mask is used

        self._unity_disc_actions = 2
        self._unity_cont_actions = 2
        self.is_action_discrete = is_action_discrete
        if is_action_discrete:
            self.action_dim = [5]
            self.action_space = [gym.spaces.Discrete(ad) for ad in self.action_dim]
        else:
            self.action_dim = [2]
            self.action_space = [gym.spaces.Box(low=-1, high=1, shape=(ad,)) for ad in self.action_dim]
        self._obs_dim = [26]
        self._global_state_dim = [28]


        self.blue_agent = Opponent(self.n_envs)
        self.blue_missile_num = 3
        # We expect <obs> in self.reset and self.step to return a list of dicts,
        # each associated to an agent. self.obs_shape and self.obs_dtype specify
        # the corresponding shape and dtype.
        # We do not consider the environment and player dimensions here!
        self.obs_shape = [dict(
            obs=self._get_obs_shape(aid),
            global_state=self._get_global_state_shape(aid),
            prev_reward=(),
            prev_action=(self.action_dim[aid],),
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
                if self.is_action_discrete:
                    self.obs_shape[aid]['action_mask'] = (self.action_space[aid].n,)
                else:
                    self.obs_shape[aid]['action_mask'] = self.action_space[aid].shape
                self.obs_dtype[aid]['action_mask'] = bool

        # The following stats should be updated in self.step and be reset in self.reset
        # The episodic score we use to evaluate agent's performance. It excludes shaped rewards
        self._win_rate = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._lose_rate = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._draw_rate = np.zeros((self.n_envs, self.n_units), dtype=np.float32)

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
        self.alive_units = np.ones((self.n_envs, 3), np.int32)

        # 红方检测到蓝方的情况
        self.detect_units = np.zeros((self.n_envs, self.n_units), bool)
        # 双方检测到的累计步数
        self.detect_steps = np.zeros((self.n_envs, self.n_units + 2), np.int32)
        # 存活步数
        self.alive_steps = np.zeros((self.n_envs, self.n_units + 2), np.int32)

        self.blue_actions = {}

        # 红方是否被锁定，-1为未被锁定，非负表示锁定的蛋，-2表示锁定的已爆（计算爆炸reward，然后重新赋值为-1
        # 锁定的蛋的距离及坐标
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
            a = np.array([[self.action_space[aid].sample() for _ in uids] for _ in range(self.n_envs)], np.float32)
            actions.append(a)
        return actions

    def reset(self):
        self._win_rate = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._lose_rate = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._draw_rate = np.zeros((self.n_envs, self.n_units), dtype=np.float32)

        self._dense_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._win_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._lose_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._detect_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._missile_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)

        self._epslen = np.zeros(self.n_envs, np.int32)

        self.alive_units = np.ones((self.n_envs, len(self._red_names[0] + self._blue_names[0])), np.int32)
        self.detect_units = np.zeros((self.n_envs, self.n_units), bool)
        self.enemy_position = np.zeros((self.n_envs, 2, 4))
        self.pre_reward = np.zeros((self.n_envs, self.n_units), np.float32)
        self._consecutive_action = np.zeros((self.n_envs, self.n_units), bool)
        self.prev_action = None

        decision_steps, terminal_steps = self.env.reset()

        return self._get_obs(decision_steps)

    def step(self, actions):
        #actions = self.conti_to_disc(actions)
        for i in range(self.frame_skip):
            if i == 0:
                self._set_blue_action(self.blue_actions)
                self._set_action(self._red_names, actions)

                for team in range(self.n_envs):
                    for b_name in self._blue_names[team]:
                        self.blue_actions[b_name][2:4] = [0, 0]

                r_main_a = actions[0]
                for team in range(self.n_envs):
                    if r_main_a[team][0] > 2:
                        r_main_a[team][0] = 0
                actions[0] = r_main_a
            else:
                self._set_blue_action(self.blue_actions)
                self._set_action(self._red_names, actions)

            reset, decision_steps, terminal_steps = self.env.step()
            if reset:
                break

        self._epslen += 1

        # TODO: Add previous actions to the observations
        agent_obs = self._get_obs(decision_steps, self.pre_reward)

        done, reward, win_r, lose_r, detect_r, missile_r = self._get_done_and_reward()

        alive_blue = np.zeros(self.n_envs)
        alive_main = np.zeros(self.n_envs)
        # 更新存活情况，用于info记录
        for i in range(self.n_envs):
            if done[i]:
                alive_main[i] = decision_steps[self._red_names[i][0]].obs[1][0][1]
                alive_blue[i] = 0

                for name in self._blue_names[i]:
                    if decision_steps[name].obs[1][0][1] == 1:
                        alive_blue[i] += 1
            else:
                for j in self.alive_units[i][-self.n_blue_main:]:
                    if j == 1:
                        alive_blue[i] += j
                alive_main[i] = 1 if self.alive_units[i][0] == 1 else 0

        for i in range(self.n_envs):
            if self._epslen[i] > self.max_episode_steps:
                done[i] = True
                print('max steps')
                self.max_steps_end += 1
                self._draw_rate[i] += 1

        discount = 1 - done  # we return discount instead of done

        assert reward.shape == (self.n_envs, self.n_units), reward.shape
        assert discount.shape == (self.n_envs,), discount.shape
        # obtain ndarrays of shape (n_envs, n_units)
        # rewards = np.tile(reward.reshape(-1, 1), (1, self.n_units))
        rewards = reward
        discounts = np.tile(discount.reshape(-1, 1), (1, self.n_units))

        self._dense_score += rewards
        assert self._dense_score[0] < 2, self._dense_score
        self._win_score += win_r
        self._lose_score += lose_r
        self._detect_score += detect_r
        self._missile_score += missile_r

        # self._score += np.where(discounts, 0, rewards > 0)  # an example for competitive games
        reset = done
        assert reset.shape == (self.n_envs,), reset.shape
        resets = np.tile(reset.reshape(-1, 1), (1, self.n_units))

        # if self.prev_action is not None:
        #     for tt in range(self.n_envs):
        #         for aid in range(self.n_agents):
        #             for id, uid in enumerate(self.aid2uids[aid]):
        #                 pa = self.prev_action[aid][tt][uid - aid]
        #                 a = actions[aid][tt][uid - aid]
        #                 self._consecutive_action[tt][uid] = (pa == a)
        self.prev_action = actions

        self._info = [dict(
            #win_game=self.win_games,
            #lose_game=self.lose_games,
            #no_missile_end_game=self.no_missile,
            #max_steps_end_game=self.max_steps_end,
            #total_game=self.total_game,
            score=self._win_rate[i].copy(),
            win_rate=self._win_rate[i].copy(),
            lose_rate=self._lose_rate[i].copy(),
            draw_rate=self._draw_rate[i].copy(),
            dense_score=self._dense_score[i].copy(),
            #win_score=self._win_score[i].copy(),
            #lose_score=self._lose_score[i].copy(),
            #detect_score=self._detect_score[i].copy(),
            #missile_score=self._missile_score[i].copy(),
            epslen=self._epslen[i],
            game_over=discount[i] == 0,
            #consecutive_action=self._consecutive_action[i].copy(),
            left_missile_red=self.red_missile_left[i],
            left_missile_blue0=self.blue_missile_left[i][0],
            left_missile_blue1=self.blue_missile_left[i][1],
            alive_blue=alive_blue[i],
            alive_main=alive_main[i],
            main_detect_steps=self.detect_steps[i][2],
            #ally0_detect_steps=self.detect_steps[i][3],
            #ally1_detect_steps=self.detect_steps[i][4],
            #ally2_detect_steps=self.detect_steps[i][5],
            #ally3_detect_steps=self.detect_steps[i][6],
            main_alive_steps=self.alive_steps[i][2],
            #ally_alive_steps=self.alive_steps[i][3:].mean(),
            blue_detect_main=self.detect_steps[i][0],
            blue0_alive_steps=self.alive_steps[i][0],
            blue1_alive_steps=self.alive_steps[i][1]
        ) for i in range(self.n_envs)]

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

                self._win_rate[i] = np.zeros(self.n_units, np.float32)
                self._lose_rate[i] = np.zeros(self.n_units, np.float32)
                self._draw_rate[i] = np.zeros(self.n_units, np.float32)

                self._epslen[i] = 0

                self.alive_units[i] = np.ones(len(self._red_names[i] + self._blue_names[i]), np.int32)
                self.detect_units[i] = np.zeros((self.n_units,), bool)
                self.detect_steps[i] = np.zeros(self.n_units + 2)
                self.alive_steps[i] = np.zeros(self.n_units + 2)
                #self.flying_missile[i] = np.ones(2, np.int32)
                #self.grid_map[i] = np.zeros((10, 10), np.int32)
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

    def _get_names(self, env):
        teams = {}
        for name in env.get_behavior_names():
            team = name[name.index('team'):]
            if teams.__contains__(team) is False:
                teams[team] = [name]
            else:
                teams[team].append(name)

        red_names = []
        blue_names = []

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
            red_names.append([name for name in sorted(v) if name.startswith('red')])
            blue_names.append([name for name in sorted(v) if name.startswith('blue')])

        return red_names, blue_names

    def _get_obs(self, decision_step, reward=None, action=None):
        def extract_obs(names):
            team_obs = [
                [decision_step[name].obs for name in team]
                for team in names
            ]
            return team_obs

        obs = [{} for _ in range(self.n_agents)]

        red_obs = extract_obs(self._red_names)
        blue_obs = extract_obs(self._blue_names)

        all_obs = self._process_obs(red_obs, blue_obs, reward, action)

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

    def _process_obs(self, red_info, blue_info, reward, action):
        n_state = 9
        self.blue_actions = {}
        red_observation = []
        self.blue_missile_left = np.zeros((self.n_envs, self.n_blue_main), np.int32)
        self.red_missile_left = np.zeros((self.n_envs, self.n_red_main), np.int32)

        # missile_end记录蛋是否已爆
        self.missile_end = np.zeros((self.n_envs, self.n_blue_main + self.n_red_main))
        self.detect_units = np.zeros((self.n_envs, self.n_units), bool)

        for team in range(self.n_envs):
            # 获取红方主僚机的射线信息、状态信息、导弹信息
            red_main_ray_info = np.array([i[0] for i in red_info[team][:self.n_red_main]]).squeeze(1)
            red_state_info = np.array([i[1][:, :n_state] for i in red_info[team]]).squeeze(1)
            red_missile_info = np.array([i[1][:, n_state:] for i in red_info[team][:self.n_red_main]]).squeeze(1)

            # 获取蓝方射线信息、状态信息、导弹信息
            blue_ray_info = np.array([i[0] for i in blue_info[team]]).squeeze(1)
            blue_state_info = np.array([i[1][:, :n_state] for i in blue_info[team]]).squeeze(1)
            blue_missile_info = np.array([i[1][:, n_state:] for i in blue_info[team]]).squeeze(1)

            # 根据状态信息获取获取各个飞机相互之间的相对坐标、距离、角色等
            all_states = np.concatenate((red_state_info, blue_state_info), axis=0)
            positions = self._get_all_positions(all_states)
            rel_pos = self._get_rel_positions(positions)
            plane_infos = self._get_all_plane_info(team, all_states)

            assert rel_pos.shape == (self.n_planes, self.n_planes, 2), rel_pos.shape
            assert len(plane_infos) == self.n_planes, len(plane_infos)

            self.red_missile_left[team] = red_missile_info[:, 1]
            self.blue_missile_left[team] = blue_missile_info[:, 1]

            observations = {}
            all_obs, all_global_state, all_alive, all_action_mask = [], [], [], []

            # 遍历红方飞机，计算observation
            for i, name in enumerate(self._red_names[team]):
                alive = red_state_info[i][1]
                self.alive_units[team][i] = alive if self.alive_units[team][i] != -1 else -1
                vel = red_state_info[i][4:6]
                direction = red_state_info[i][7:9]
                left_missile = red_missile_info[i][1] if i < self.n_red_main else 0
                nearest_wall = np.array([min(abs(red_state_info[i][2]-150), abs(red_state_info[i][2]+150),
                                             abs(red_state_info[i][3]-50), abs(red_state_info[i][3]+50))])

                other_info = [info for j, info in enumerate(plane_infos) if j != i]
                other_state_info = copy.deepcopy(other_info)

                other_position = [rp for j, rp in enumerate(rel_pos[i]) if j != i]

                for j, blue_info in enumerate(other_position[-self.n_blue_main:]):
                    blue_info = self.enemy_position[team, j, :2]  # 用env维护的蓝方位置信息替换真实位置信息

                other_distance = np.array([np.linalg.norm(j) for j in other_position])
                # 获取是否探测到蓝方
                n_enemies = self._get_detect_blues(red_main_ray_info[i], i)

                # rewards[i][v] += sum(detect_flag) * self.reward_setting['DETECT_REWARD']
                detect_flag = np.zeros(self.n_blue_main, np.float32)
                detect_flag[0:n_enemies - 1] = 1 if n_enemies != 0 else 0

                self.detect_units[team][i] += n_enemies
                self.detect_steps[team][i] += sum(detect_flag)
                self.alive_steps[team][i] += alive == 1

                # for bid in range(2):
                #     self.enemy_position[team][bid] = blue_state_info[bid, 2:6]
                # 蓝方位置信息
                for bid in range(2):
                    if detect_flag[bid] == 1:
                        self.enemy_position[team][bid] = blue_state_info[bid, 2:6]
                    else:
                        # TODO: maintain an additional counter
                        if self._epslen[team] % 7 == 0:
                            self.enemy_position[team][bid] = blue_state_info[bid, 2:6] + np.random.randn(4)
                        else:
                            self.enemy_position[team][bid][0:2] = self._predict_position(
                                self.enemy_position[team][bid][0:2],
                                self.enemy_position[team][bid][2:4],
                                self._epslen[team] % 7)
                # 锁定主机和自己的最近的蛋
                me_missile = self._get_target_missile(blue_missile_info, team, i, positions[i])
                # if me_missile != 0:
                #     assert me_missile < MAX_LEN, me_missile
                #     self.missile_dis[team][v] -= me_missile
                blue_missile_left = blue_missile_info[:, 1] - 2
                obs = [
                    vel,#0, 2
                    direction,#2, 2
                    np.array([left_missile]),#4, 1
                    other_distance, # 5, 2
                    detect_flag, # 7, 2
                    me_missile, # 9, 1
                    nearest_wall, # 10, 1
                    *other_state_info, # 11, 10
                    *other_position # 21, 4
                ]
                # other_state_info 12, 10
                # print(f'vel: {obs[0:2]}\ndirection: {obs[2:4]}\nleft missile')
                global_state = [
                    vel, #0, 2
                    direction, #2, 2
                    np.array([left_missile]),#4, 1
                    other_distance, # 5, 2
                    detect_flag,# 7, 2
                    me_missile,# 9, 1
                    nearest_wall, # 10, 1
                    *other_state_info,  # 11, 10
                    *other_position, # 21, 4
                    blue_missile_left, # 25, 2
                ]

                if i < self.n_red_main:
                    # if self._epslen[team] < 10:
                    #     if red_missile_info[-2] == 1 or red_missile_info[-1] ==1:
                    #         print(self._epslen[team])
                    #         print(red_missile_info[-2:])

                    # 遍历主机的导弹信息，用于控制连续发蛋和已爆的蛋数
                    # for j in range(9, 37, 9):
                    #     #
                    #     #     if self.flying_missile[team][int(red_missile_info[j])] == 1:
                    #     if red_missile_info[i, j + 1] == 1:
                    #         missile_mask[int(red_missile_info[i, j])] = 0
                    #         #self.flying_missile[team][int(red_missile_info[j])] = 0
                    #     if red_missile_info[i, j + 1] == 2:
                    #         #self.flying_missile[team] = 1
                    #         self.missile_end[team][2] += 1

                    action_mask = np.ones(self.action_dim[self.uid2aid[i]], bool)
                    # 根据可打击目标和飞行中的蛋计算action mask
                    # TODO: remove the blue order
                    if self.is_action_discrete:
                        action_mask[-self.n_blue_main:] = red_missile_info[i][-self.n_blue_main:]

                    all_action_mask.append(action_mask)

                    obs.append(np.array([red_missile_info[i, 1]]))
                    global_state.append(np.array([red_missile_info[i, 1]]))
                else:
                    all_action_mask.append(np.ones(self.action_dim[self.uid2aid[i]], bool))

                obs = np.concatenate(obs)
                global_state = np.concatenate(global_state)
                if not alive:
                    obs = np.zeros_like(obs)
                all_obs.append(obs)
                # print('raw velocity: ', vel)
                # print('raw direction: ', direction)
                # print(f'obstmp: {obs}')
                # print(f'split res vel: {obs[0:2]}\ndirection: {obs[2:4]}\nleft missile: {obs[4:5]}\n'
                # f'other_distance:{obs[5:7]}\ndetect_flag:{obs[7:9]}\nme_missile:{obs[9:10]}\n'
                # f'nearest_wall:{obs[10:11]}\nother_state_info:{obs[11:21]}\nother_position:{obs[21:25]},red_missile_info:{obs[25]}')
                
                all_global_state.append(global_state)
                all_alive.append(alive)

            observations['obs'] = all_obs
            observations['global_state'] = all_global_state
            observations['life_mask'] = all_alive
            observations['action_mask'] = all_action_mask
            observations['prev_reward'] = np.zeros(self.n_units, np.float32) if reward is None else reward[team]
            observations['prev_action'] = np.zeros((self.n_units, self.action_dim[team]),
                                                   np.float32) if action is None else action[team]

            red_observation.append(observations)

            # Update blue
            blue_observation = [blue_ray_info, blue_state_info, blue_missile_info,
                                red_state_info, red_missile_info]
            # 获取蓝方探测到红方的数据、蓝方的决策动作
            detect_red, b_a = self.blue_agent.choose_action(team, blue_observation)
            self.detect_steps[team][0] += 1 if detect_red[0] != 0 else 0
            self.detect_steps[team][1] += 1 if detect_red[1] != 0 else 0

            self.blue_actions[self._blue_names[team][0]] = b_a[0]
            self.blue_actions[self._blue_names[team][1]] = b_a[1]
            assert len(self.blue_actions) == 2 * (team + 1), self.blue_actions

            # 更新蓝方的存活数据
            self._update_blue_alive(team, blue_state_info)

            # 更新蓝方蛋的生命周期
            self._update_missile_life(blue_missile_info, team)

        return red_observation

    def _update_missile_life(self, blue_missile_info, team):
        for id, missile_info in enumerate(blue_missile_info):
            for missle_left in range(2, len(missile_info) - 5 - 9 + 1, 9):
                if missile_info[missle_left + 8] == 2:
                    self.missile_end[team][id+1] += 1


    def _update_blue_alive(self, team, blue_state_info):
        for id, blue in enumerate(self._blue_names[team]):
            if blue_state_info[id][1] == 1:
                self.alive_steps[team][id + 1] += 1
            if blue_state_info[id][1] == 0 and self.alive_units[team][id+1] == 1:
                self.alive_units[team][id+1] = 0

    def _get_all_positions(self, all_states):
        pos = np.array(all_states[:, 2:4])
        return pos

    def _get_rel_positions(self, pos):
        rel_pos = np.expand_dims(pos, 0) - np.expand_dims(pos, 1)
        return rel_pos

    def _get_all_plane_info(self, team, all_states):
        ret = []
        for i, name in enumerate(self._red_names[team] + self._blue_names[team]):
            state = all_states[i]
            role = self._get_plane_role(name)
            #velocity = list(state[4:6])
            #direction = list(state[7:9])
            alive = [1, 0] if state[1] == 1 else [0, 1]

            #ret.append(role + velocity + direction + alive)
            ret.append(role + alive)
        return ret

    def _get_plane_role(self, name):
        if name.startswith('red_main'):
            return [1, 0, 0]
        if name.startswith('red_sup'):
            return [0, 1, 0]
        if name.startswith('blue'):
            return [0, 0, 1]

    def _predict_position(self, last_pos, vel, t):
        return last_pos + 0.1 * vel * t * self.frame_skip

    def _get_detect_blues(self, ray_info, agent_id):
        if agent_id < self.n_red_main:
            tag_num = 4
            ray_length = self._red_main_ray_dim
            enemy_index = 1

        n_enemies = 0
        for i in range(enemy_index, ray_length, tag_num + 2):
            n_enemies += ray_info[i] == 1

        return n_enemies

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
        distance = 50
        for id, missile in enumerate(enemy_missile):
            for j in range(2, len(missile) - 5 - 9 + 1, 9):
                if missile[j + 7] == target and missile[j + 8] == 1:
                    m_dis = np.linalg.norm(missile[j:j + 2] - position)
                    if distance == 0 or m_dis < distance:
                        distance = m_dis

        return np.array([distance])

    def _conti_to_disc(self, actions):
        move = actions[0]
        shot = actions[1]
        #assert -1 <= move <= 1, move
        #assert -1 <= shot <= 1, shot

        if move < -1/3 and shot < -1/3:
            return 0
        if move < 1/3 and shot < -1/3:
            return 1
        if move <= 1 and shot < -1/3:
            return 2
        if shot < 1/3:
            return 3
        else:
            return 4




    def _set_blue_action(self, actions):
        for name, action in actions.items():
            action_tuple = self.env.get_action_tuple()
            action_tuple.add_discrete(np.array(action[2:4]).reshape(1, 2))
            action_tuple.add_continuous(np.array(action[0:2]).reshape(1, 2))
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
                    action = np.array([1, -1], np.float32).reshape(1, 2), np.array([0, 0], np.int32).reshape(1, 2)
                elif action == 2:
                    action = np.array([1, 1], np.float32).reshape(1, 2), np.array([0, 0], np.int32).reshape(1, 2)
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
                if self.is_action_discrete:
                    action = actions[tid][uid]
                else:
                    action = self._conti_to_disc(actions[tid][uid])
                cont_action, dist_action = to_unity_action(n, action)
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
            if self.alive_units[i][-1] != 1 and self.alive_units[i][-2] != 1 and self.alive_units[i][0] == 1 and \
                    (self.missile_end[i][1] + self.blue_missile_left[i][0]) == \
                    (self.missile_end[i][2] + self.blue_missile_left[i][1]) == self.blue_missile_num:
                done[i] = True
                reward[i] += self.reward_config['blue_dead_reward']
                win_r[i] += self.reward_config['blue_dead_reward']
                print('red win')
                self.win_games += 1
                self._win_rate[i] += 1

                if self.alive_units[i][-1] == 0:
                    self.alive_steps[i][-1] = self._epslen[i]
                if self.alive_units[i][-2] == 0:
                    self.alive_steps[i][-2] = self._epslen[i]
            # 红方主机死亡
            elif self.alive_units[i][0] != 1:
                done[i] = True
                reward[i] += self.reward_config['main_dead_reward']
                lose_r[i] += self.reward_config['main_dead_reward']
                print('red lose')
                self.alive_steps[i][0] = self._epslen[i]
                self.lose_games += 1
                self._lose_rate[i] += 1

            # 蓝方主机都活着，但是红蓝双方所有蛋都打完
            if self.alive_units[i][-2] == 1 and self.alive_units[i][-1] == 1:
                if sum(self.missile_end[i]) == 10-4:
                    done[i] = True
                    self.no_missile += 1
                    self._draw_rate[i] += 1

                    # print('no missile')
            else:
                # 处理蓝方死掉一个飞机，其他活着的飞机都没蛋的情况
                if self.alive_units[i][-1] != 1 and self.missile_end[i][0] == 4 \
                        and self.missile_end[i][-2] == 1:
                    done[i] = True
                    # print('no missile')
                    self.no_missile += 1
                    self._draw_rate[i] += 1

                if self.alive_units[i][-2] != 1 and self.missile_end[i][0] == 4 \
                        and self.missile_end[i][-1] == 1:
                    done[i] = True
                    # print('no missile')
                    self.no_missile += 1
                    self._draw_rate[i] += 1

            if done[i] == 0:
                for j in range(len(self._blue_names[0])):
                    if self.alive_units[i][j+1] == 0:
                        #reward[i] += self.reward_config['blue_dead_reward']
                        win_r[i] += self.reward_config['blue_dead_reward']
                        self.alive_units[i][j+1] = -1
                        self.alive_steps[i][j+1] = self._epslen[i]

                #for j in range(len(self._red_names[0])):
                    #assert sum(self.detect_units[i][j]) <= 2
                    # if self.missile_end[i][2] != 4:
                    #reward[i][j] += self.reward_config['detect_reward'] * sum(self.detect_units[i][j])
                    #detect_r[i][j] += self.reward_config['detect_reward'] * sum(self.detect_units[i][j])

                    # if self.missile_locked[i][j] == -2:
                    #     bomb_dis = np.linalg.norm(self.missile_position[i][j])
                    #     if bomb_dis > MAX_LEN:
                    #         bomb_dis = MAX_LEN - 1
                    #     # reward[i][j] -= 0.05 * np.log(MAX_LEN - bomb_dis)
                    #     missile_r[i][j] -= 0.05 * np.log(MAX_LEN - bomb_dis)
                    #
                    #     self.missile_locked[i][j] = -1

                # for g_x in range(10):
                #     for g_z in range(10):
                #         if self.grid_map[i][g_x][g_z] == 1:
                #             # reward[i] += self.reward_config['grid_reward']
                #             self.grid_map[i][g_x][g_z] = -1

        self.pre_reward = reward.copy()
        return done, reward, win_r, lose_r, detect_r, missile_r


""" Test Code """
if __name__ == '__main__':
    config = dict(
        env_name='dummy',
        uid2aid=[0],
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


    def print_dict_info(d, prefix=''):
        for k, v in d.items():
            if isinstance(v, dict):
                print(f'{prefix} {k}')
                print_dict_info(v, prefix + '\t')
            elif isinstance(v, tuple):
                # namedtuple is assumed
                print(f'{prefix} {k}')
                print_dict_info(v._asdict(), prefix + '\t')
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
        print_dict_info(o, f'\tagent{i}')

    for k in range(1, 50000):
        # env.env.reset_envs_with_ids([2])
        actions = env.random_action()
        print(f'Step {k}, random actions', actions)
        observations, rewards, dones, reset = env.step(actions)
        print(f'Step {k}, observations')
        for i, o in enumerate(observations):
            print_dict_info(o, f'\tagent{i}')
        print(f'Step {k}, rewards', rewards)
        print(f'Step {k}, dones', dones)
        print(f'Step {k}, reset', reset)
        info = env.info()
        print(f'Step {k}, info')
        for aid, i in enumerate(info):
            print_dict(i, f'\tenv{aid}')
