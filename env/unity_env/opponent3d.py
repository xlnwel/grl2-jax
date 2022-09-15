import math
import time
import numpy as np
import random
from queue import Queue

MAX_RAY_DIS = 60
SHOT_DIS_MAIN = 40
SHOT_DIS_ALLY = 30
SCAPE_MISSILE_DIS = 60
CHASE_LEN = 1


class Opponent3d(object):

    def __init__(self, n_copies):
        self.blue_name_ids = {'E0_Blue_0?team=0': 0, 'E0_Blue_1?team=0': 1}
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
        self.position = blue_state[:, 2:5]
        self.vel = blue_state[:, 5:8]
        self.angle_vel = blue_state[:, 8:11]
        self.angle = blue_state[:, 11:14]
        self.direction = blue_state[:, 14:17]
        self.overload = blue_state[:, 17]
        self.oil = blue_state[:, 18]
        self.locked = blue_state[:, 19]
        self.shoted = blue_state[:, 20]

        self.detect_red = np.zeros((2, 5))
        self.detect_red_main_pos = red_state[0, 2:5]
        self.detect_red_ally_pos = red_state[1:, 2:4]
        self.lock_red_id = self.ray_front[:, 1]

        self.left_missile = blue_missile[:, 1]
        self.target_list = blue_missile[:, -5:]
        self.flying_missile = np.zeros((2, 4))

        for i in range(len(self.blue_name_ids)):
            for j in range(4, len(self.ray_front[i]), 4):
                if self.ray_front[i][j] == 1:
                    self.detect_red[i][int(j / 4) - 1] = 1
            assert blue_missile[i].shape == (88,), blue_missile[i]
            for j in range(4):
                if blue_missile[i][45 + 14 * j] == 1:
                    self.flying_missile[i][j] = 1

            # self.detect_red_ally[i].sort()

    def choose_action(self, team, obs):
        blue_ray_info = obs[0]
        blue_state_info = obs[1]
        blue_missile_info = obs[2]
        red_state_info = obs[3]

        self.update_state([blue_state_info, blue_ray_info, blue_missile_info, red_state_info])

        # 更新红方十步前的位置信息
        if self.history_pos[team].qsize() == 10:
            self.history_pos[team].get()
        self.history_pos[team].put(self.detect_red_main_pos)

        detect_red = [0, 0]
        move_action = np.zeros((2, 3))
        radar_action = np.ones((2, 1))
        shot_action = np.zeros((2, 1))
        for i in range(2):
            if self.lock_red_id[i] != -1 and np.count_nonzero(self.flying_missile[i]) == 0:
                shot_action[i] = 1
            if self.detect_red[i][0] != 0:
                radar_action[i] = 1
            else:
                for j in range(1, 5):
                    if self.detect_red[i][j] != 0:
                        radar_action[i] = j + 1
        move_action[0] = np.array([0, 0, 0])
        move_action[1] = np.array([0, 0, 0])
        for i in range(2):
            if self.position[i][0] < -15 or self.position[i][0] > 15 or \
                    self.position[i][2] < -78 or self.position[i][2] > 78:
                move_action[i] = np.array([0, -1, 0])
            if self.position[i][1] > 12:
                move_action[i] = np.array([0, 0, 0.08])
            if self.position[i][1] < 3:
                move_action[i] = np.array([0, 0, -0.08])
        if self.locked[0] or self.shoted[0]:
            move_action[0] = np.array([0, -1, 0])
        if self.locked[1] or self.shoted[1]:
            move_action[1] = np.array([0, -1, 0])
        return move_action, radar_action, shot_action

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
        if dis < SHOT_DIS_MAIN + 10 * self.left_missile[bid] and isMain:
            # if dis < SHOT_DIS_MAIN and isMain:
            shot_id = 0
            shot = 1

        if dis < SHOT_DIS_ALLY and isMain is False and self.left_missile[bid] > 1:
            shot_id = id + 1
            shot = 1

        if angle < 0:
            turn = 1
        else:
            turn = -1

        if self.flying_missile[bid].any():
            if shot_id == 0 and shot == 1 and self.flying_missile[bid][0] == 0:
                return np.array([1, turn, shot_id, shot])
            else:
                return np.array([1, turn, 0, 0])
        else:
            # if shot == 1:
            #     print(bid, 'shot', shot_id)
            if self.flying_missile[1 - bid][shot_id] == 1 and shot_id != 0:
                return np.array([1, turn, 0, 0])
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

    def avoid_missile(self, position, dir, miss_info):
        v1 = dir
        v2 = miss_info[-2:]
        v_norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if v_norm == 0:
            return np.array([1, 0])

        cos_v = np.dot(v1, v2) / v_norm

        if cos_v >= 0:
            # return np.array([1, 1])
            return np.array([1, 0])
        else:
            return np.array([1, 1])
