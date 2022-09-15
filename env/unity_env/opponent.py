import math

import numpy as np
import random
from queue import Queue

MAX_RAY_DIS = 60
SHOT_DIS_MAIN = 80
SHOT_DIS_ALLY = 30
SCAPE_MISSILE_DIS = 60
CHASE_LEN = 5


class Opponent(object):

    def __init__(self, n_copies):
        self.blue_name_ids = {'blue_0?team=0': 0, 'blue_1?team=0': 1}
        self.n_copies = n_copies
        self.tag_num = 4
        self.history_pos = [Queue(10) for i in range(n_copies)]
        self.chase_steps = [0] * n_copies

    def update_state(self, info):
        self.ray_front = info[1]
        blue_state = info[0]
        red_state = info[-1]
        blue_missile = info[2]
        self.is_alive = blue_state[:, 1]
        self.position = blue_state[:, 2:4]
        self.vel = blue_state[:, 4:6]
        self.angle = blue_state[:, 6]
        self.direction = blue_state[:, 7:9]
        self.detect_red_main = np.zeros((2, 1))
        self.detect_red_ally = [[], []]
        self.detect_red_main_pos = red_state[0, 2:4]
        self.detect_red_ally_pos = red_state[1:, 2:4]
        self.detect_wall = np.zeros((2, 1))
        self.total_missile = blue_missile[:, 0]
        self.left_missile = blue_missile[:, 1]
        self.target_list = blue_missile[:, -5:]
        self.flying_missile = np.zeros((2, 5), np.bool)
        self.detect_missile = np.zeros((2, 1))

        for i in range(len(self.blue_name_ids)):
            for j in range(0, len(self.ray_front[i]), self.tag_num + 2):
                if self.ray_front[i][j] == 1:
                    self.detect_wall[i] = True
                    break

            for j in range(1, len(self.ray_front[i]), self.tag_num + 2):
                if self.ray_front[i][j] == 1:
                    self.detect_red_main[i] = self.ray_front[i][j + 4]
                    # self.detect_red_main_pos[i] = self._calculate_detect_position(j//(TAG_NUM+2), self.detect_red_main[i])

            for j in range(2, len(self.ray_front[i]), self.tag_num + 2):
                if self.ray_front[i][j] == 1:
                    self.detect_red_ally[i].append(self.ray_front[i][j + 3])

            for j in range(3, len(self.ray_front[i]), self.tag_num + 2):
                if self.ray_front[i][j] == 1:
                    self.detect_missile[i] = 1
                    break

            for j in range(9, 28, 9):
                if blue_missile[i][j + 1] == 1:
                    self.flying_missile[i][int(blue_missile[i][j])] = True

            # self.detect_red_ally[i].sort()

    def choose_action(self, team, obs):
        blue_ray_info = obs[0]
        blue_state_info = obs[1]
        blue_missile_info = obs[2]
        red_state_info = obs[3]
        red_missile_info = obs[4]

        self.update_state([blue_state_info, blue_ray_info, blue_missile_info, red_state_info])

        # 更新红方十步前的位置信息
        if self.history_pos[team].qsize() == 10:
            self.history_pos[team].get()
        self.history_pos[team].put(self.detect_red_main_pos)

        detect_red = [0, 0]
        if sum(self.detect_red_main) != 0:
            if self.chase_steps[team] != 0:
                actions = [np.array([1, 0, 0, 0]), np.array([1, 0, 0, 0])]
            if self.chase_steps[team] % CHASE_LEN == 0:
                actions = self.chase_main()
                self.chase_steps[team] = 0
            detect_red[0] += 1
            self.chase_steps[team] += 1
        else:
            # 没探测到主机，探测到了无人机
            if (sum(self.detect_red_ally[0]) + sum(self.detect_red_ally[1])) != 0:
                if self.chase_steps[team] != 0:
                    actions = [np.array([1, 0, 0, 0]), np.array([1, 0, 0, 0])]
                if self.chase_steps[team] % CHASE_LEN == 0:
                    actions = self.chase_ally()
                    self.chase_steps[team] = 0
                detect_red[1] += 1
                self.chase_steps[team] += 1
            else:
                # 未发现红方
                if self.chase_steps[team] != 0:
                    actions = [np.array([1, 0, 0, 0]), np.array([1, 0, 0, 0])]
                if self.chase_steps[team] % CHASE_LEN == 0:
                    self.chase_steps[team] = 0

                    # if self.search_step[team] == 10:
                    # self.search_step[team] = 0
                    target = self.history_pos[team].get()
                    actions = np.array(
                        [self._chase(target - self.position[0], self.direction[0], isMain=True, id=0, bid=0),
                         self._chase(target - self.position[1], self.direction[1], isMain=True, id=0, bid=1)])

                    # else:
                    #     actions = self.search_grid()
                self.chase_steps[team] += 1

        red_missile = red_missile_info

        distance = [1000, 1000]
        chaser_m = [None, None]
        # 记录打击自己的红方蛋的信息，进行避蛋
        for j in range(2, len(red_missile) - 2 - 9 + 1, 9):
            if red_missile[j + 8] == 1:
                d = min(distance[int(red_missile[j + 7])], np.linalg.norm(red_missile[j:j + 2] - self.position))
                distance[int(red_missile[j + 7])] = d
                chaser_m[int(red_missile[j + 7])] = red_missile[j:j + 7]
        for index, dd in enumerate(distance):
            if dd < SCAPE_MISSILE_DIS:
                actions[index][0:2] = self.avoid_missile(self.position[index], self.direction[index], chaser_m[index])
        # for index, m in enumerate(self.detect_missile):
        #     if m == 1:
        #         actions[index] = self.avoid_missile()

        # for blue_id in range(len(self.blue_name_ids.keys())):
        #     for enemy_id in range(5):
        #         if self.target_list[blue_id][enemy_id] == 1 and self.left_missile[blue_id] > 0 :
        #             actions[blue_id][3] = 1
        #             actions[blue_id][2] = enemy_id

        return detect_red, np.array(actions)

    # TODO search the map
    def search_grid(self):
        actions = [
            np.array(random.choice([[1, -1, 0, 0], [1, 1, 0, 0]])) if self.detect_wall[0] else np.array([1, 0, 0, 0]),
            np.array(random.choice([[1, -1, 0, 0], [1, 1, 0, 0]])) if self.detect_wall[1] else np.array([1, 0, 0, 0])]

        return np.array(actions)

    def chase_main(self):
        return np.array(
            [self._chase(self.detect_red_main_pos - self.position[0], self.direction[0], isMain=True, id=0, bid=0),
             self._chase(self.detect_red_main_pos - self.position[1], self.direction[1], isMain=True, id=0, bid=1)])

    def chase_ally(self):
        actions = []
        target_pos = [None, None]
        target_index = [None, None]

        for p in range(2):
            if sum(self.detect_red_ally[p]) == 0:
                continue

            target_dis = min(self.detect_red_ally[p]) * MAX_RAY_DIS
            pos = self.position[p]
            min_dis = 1000
            for j in range(len(self.detect_red_ally_pos)):
                i = self.detect_red_ally_pos[j]
                dis_e = math.fabs(target_dis - math.sqrt(math.pow(i[0] - pos[0], 2) + math.pow(i[1] - pos[1], 2)))
                if min_dis > dis_e:
                    min_dis = dis_e
                    target_pos = i
                    target_index = j
        actions.append(
            self._chase(target_pos - self.position[0], self.direction[0], isMain=False, id=target_index, bid=0))
        actions.append(
            self._chase(target_pos - self.position[1], self.direction[1], isMain=False, id=target_index, bid=1))
        return np.array(actions)

    # TODO use continuous action
    def _chase(self, diff, vel, isMain, id, bid):
        shot_id = 0
        turn = 0
        shot = 0
        angle = self.get_clock_angle(vel, diff)
        dis = math.sqrt(math.pow(diff[0], 2) + math.pow(diff[1], 2))
        if dis < SHOT_DIS_MAIN and isMain:
            shot_id = 0
            shot = 1
        if dis < SHOT_DIS_ALLY and isMain is False and self.left_missile[bid] > 1:
            shot_id = id + 1
            shot = 1
        if angle < 0:
            turn = 1
        else:
            turn = -1

        if self.flying_missile[bid][shot_id]:
            return np.array([1, turn, 0, 0])
        else:
            return np.array([1, turn, shot_id, shot])
            # return np.array([1, turn, 0, 0])

    # 获取自己方向矢量和目标相对位置的夹角
    def get_clock_angle(self, v1, v2):
        v_norm = np.linalg.norm(v1) * np.linalg.norm(v2)

        sin_v = np.cross(v1, v2) / v_norm
        cos_v = np.dot(v1, v2) / v_norm
        sin_v = 1 if sin_v > 1 else sin_v
        cos_v = 1 if cos_v > 1 else cos_v
        sin_v = -1 if sin_v < -1 else sin_v
        cos_v = -1 if cos_v < -1 else cos_v

        rho = np.rad2deg(np.arcsin(sin_v))
        theta = np.rad2deg(np.arccos(cos_v))

        if rho < 0:
            return - theta
        else:
            return theta

    def predict_model(self):
        pass

    def _calculate_detect_position(self, index, distance):
        pass

    def avoid_missile(self, position, dir, miss_info):
        v1 = dir
        v2 = miss_info[-2:]
        v_norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if v_norm == 0:
            return np.array([1, 0])

        cos_v = np.dot(v1, v2) / v_norm

        if cos_v >= 0:
            return np.array([1, 1])
            return np.array([1, 0])
        else:
            return np.array([1, 1])



