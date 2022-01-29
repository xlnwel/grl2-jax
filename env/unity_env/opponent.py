import math

import numpy as np
import random

WIDTH = 100
HEIGHT = 100
MAX_RAY_DIS = 20
MAX_RAY_DEGREE = 30
SHOT_DIS_MAIN = 20
SHOT_DIS_ALLY = 15
SCAPE_MISSLE_DIS = 10

class Opponent(object):

    def __init__(self, n_copies):
        self.blue_name_ids = {'blue_main?team=0':0, 'blue_sup_0?team=0':1}
        self.n_copies = n_copies
        self.tag_num = 4

    def update_state(self, info):
        # self.state = obs['state']
        # blue_info = obs['blue_info']
        self.ray_front = info[1]
        blue_state = info[0]
        red_state = info[-1]
        blue_missile = info[2]
        #self.is_alive = [i[BLUE_RAY_DIM + 1: BLUE_RAY_DIM + 2] for i in blue_info]
        self.is_alive = blue_state[:, 1]
        self.position = blue_state[:, 2:4]
        self.vel = blue_state[:, 4:6]
        self.angle = blue_state[:, 7]
        self.angle_ = [(math.sin(i), math.cos(i)) for i in self.angle]
        self.detect_red_main = np.zeros((2, 1))
        self.detect_red_ally = [[],[]]
        self.detect_red_main_pos = red_state[0, 2:4]
        self.detect_red_ally_pos = red_state[1:, 2:4]
        self.detect_wall = np.zeros((2, 1))
        self.total_missile = blue_missile[:, 0]
        self.left_missile = blue_missile[:, 1]
        self.target_list = blue_missile[:, -5:]

        for i in range(len(self.blue_name_ids)):
            for j in range(0, len(self.ray_front[i]), self.tag_num+2):
                if self.ray_front[i][j] == 1:
                    self.detect_wall[i] = True
                    break

            for j in range(1, len(self.ray_front[i]), self.tag_num+2):
                if self.ray_front[i][j] == 1:
                    self.detect_red_main[i] = self.ray_front[i][j+3]
                    #self.detect_red_main_pos[i] = self._calculate_detect_position(j//(TAG_NUM+2), self.detect_red_main[i])

            for j in range(2, len(self.ray_front[i]), self.tag_num+2):
                if self.ray_front[i][j] == 1:
                    self.detect_red_ally[i].append(self.ray_front[i][j+2])

            #self.detect_red_ally[i].sort()

    def choose_action(self,env, obs):
        blue_ray_info = obs[0]
        blue_state_info = obs[1]
        blue_missile_info = obs[2]
        red_state_info = obs[3]
        red_missile_info = obs[4]

        all_actions = []
        # 探测到主机
        for i in range(self.n_copies):
            self.update_state([blue_state_info[i], blue_ray_info[i], blue_missile_info[i], red_state_info[i]])

            if sum(self.detect_red_main) != 0:
                actions = self.chase_main()
            else:
                # 没探测到主机，探测到了无人机
                if (sum(self.detect_red_ally[0]) + sum(self.detect_red_ally[1])) != 0:
                    actions = self.chase_ally()
                else: # 未发现红方
                    actions = self.search_grid()

            red_missile = red_missile_info[i]

            distance = [1000, 1000]
            for id, missile in enumerate(red_missile):
                for j in range(2, len(missile) - 2 - 9 + 1, 9):
                    if missile[j + 8] == 1:
                        d = min(distance[int(missile[j + 7])], np.linalg.norm(missile[j:j + 2] - self.position))
                        distance[int(missile[j + 7])] = d
            for index, dd in enumerate(distance):
                if dd < SCAPE_MISSLE_DIS:
                    actions[index] = self.avoid_missile(dd)

            for blue_id in range(len(self.blue_name_ids.keys())):
                for enemy_id in range(5):

                    if self.target_list[blue_id][enemy_id] == 1 and self.left_missile[blue_id] >0 :
                        actions[blue_id][3] = 1
                        actions[blue_id][2] = enemy_id

            all_actions.append(actions)
        all_actions = np.array(all_actions)
        ret = {}
        for k, v in self.blue_name_ids.items():
            ret[k] = all_actions[:, v]

        self.actions = ret

    #TODO search the map
    def search_grid(self):
        actions = [np.array(random.choice([[-1, 1, 0, 0], [1, 1, 0, 0]])) if self.detect_wall[0] else np.array([1, 0, 0, 0]),
                   np.array(random.choice([[-1, 1, 0, 0], [1, 1, 0, 0]])) if self.detect_wall[0] else np.array([1, 0, 0, 0])]
        #actions = [np.array(random.choice([1, 2])) if self.detect_wall[0] else np.array(1),
        #           np.array(random.choice([1, 2])) if self.detect_wall[0] else np.array(1)]

        return np.array(actions)

    def chase_main(self):
        return np.array([self._chase(self.position[0], self.detect_red_main_pos, isMain=True, id=0),
                self._chase(self.position[1], self.detect_red_main_pos, isMain=True, id=0)])

    def chase_ally(self):
        actions = []
        target_pos = [None, None]
        target_index = [None, None]

        for p in range(2):
            if sum(self.detect_red_ally[p]) == 0:
                continue

            target_dis = min(self.detect_red_ally[p]) * MAX_RAY_DIS
            pos = self.position[p]
            for j in range(len(self.detect_red_ally_pos)):
                i = self.detect_red_ally_pos[j]
                dis = math.sqrt(math.pow(i[0] - pos[0], 2) + math.pow(i[1] - pos[1], 2))
                if math.fabs(target_dis-dis) < 1:
                    target_pos = i
                    target_index = j
        actions.append(self._chase(self.position[0], target_pos, isMain=False, id=target_index))
        actions.append(self._chase(self.position[1], target_pos, isMain=False, id=target_index))
        return np.array(actions)

    #TODO use continuous action
    def _chase(self, pos1, pos2, isMain, id):
        shot_id = 0
        turn = 0
        shot = 0
        diff = (pos1[0]-pos2[0], pos1[1]-pos2[1])
        dis = math.sqrt(math.pow(diff[0], 2) + math.pow(diff[1], 2))
        if dis < SHOT_DIS_MAIN and isMain:
            shot_id = 0
            shot = 1
        if dis < SHOT_DIS_ALLY and isMain is False:
            shot_id = id + 1
            shot = 1
        if diff[0] > 0:
            turn = 1
        else:
            turn = -1
        return np.array([1, turn, shot_id, shot])

    def predict_model(self):
        pass

    def _calculate_detect_position(self, index, distance):
        pass

    #TODO scape from missile
    def avoid_missile(self, missle_dis):
        return np.array(random.choice([[-1, 1, 0, 0], [1, 1, 0, 0]]))
        #return np.array([random.choice([[[-1, 1, 0, 0]], [1, 1, 0, 0]])])



