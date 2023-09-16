from sqlite3 import NotSupportedError
import gym
from gym import spaces
import numpy as np
from copy import copy
# from mushroom_rl.utils.viewer import Viewer

class BypassEnv(gym.Env):
  '''
  This is a simple environment where the agent needs to bypass the water and reaches the goal as soon as possible
  =============
  O  X  W   O
  E  W   W  E
  O   W   O
  =============

  version 0 is a 5*13 grid with four pools of water

  X denotes the agent, "=" or "O" denotes obstacle, W denots water, E in the leftmost denotes entrance, E in the rightmost denotes exits
  '''
  def __init__(self) -> None:
    super().__init__()
    # coordinate + num_of_cross_water
    self.map = np.zeros((5, 13), dtype=np.int)
    self.state = []
    self.history_step = 5
    self.history_pos = []
    self.current_pos = [2, 0]  # goal locates in [2,12]
    self.goal = [2, 12]
    self.num_in_water = 0.0
    self.cur_step = 1
    # reset variables' default value
    self.reset()

    # self._viewer = rendering.Viewer(600, 400)
    self._scale = 2
    # self._viewer = Viewer(self._scale*300, self._scale*140, self._scale*300, self._scale*140, background=(255,255,255))  # NOTE: the coordinate is YoX, Y axis towards the right, X axis towards the up

    self.MAX_STEP = 100
    self.max_episode_steps = self.MAX_STEP
    self.observation_space = spaces.Box(low=0, high=1, shape=(self.history_step*2+4,), dtype=np.float32)
    self.action_space = spaces.Discrete(5)   # U, D, L, R, S

  def reset(self):
    self.map[1,0], self.map[3,0], self.map[1,12], self.map[3,12] = 1, 1, 1, 1  # obstacle
    self.map[0,:] = 1  # obstacle
    self.map[4,:] = 1
    self.map[1,6], self.map[3,6], self.map[2,3], self.map[2,9] = 2, 2, 2, 2  # water
    self.num_in_water = 0.0  # this will be divided by 4 as policy input
    self.history_pos = []  # will be followed by current pos
    self.current_pos = [2, 0]  # goal locates in [2,12]
    self.state = copy(self.current_pos)  # state = (current pos, num of water, dis2goal), not the obs for agent
    self.state[0] /= 4.0  # norm to 0~1
    self.state[1] /= 12.0
    self.cur_step = 1
    self.state.append(self.num_in_water)
    L1_dis_to_goal = np.abs(self.goal[0]-self.current_pos[0])+ np.abs(self.goal[1]-self.current_pos[1])
    self._prev_dis = L1_dis_to_goal/10.0
    self.state.append(L1_dis_to_goal/10.0)
    obs = np.zeros(self.history_step*2).tolist()
    obs.extend(self.state)
    return obs

  def step(self, act):
    rew = -0.01  # time punishment
    done = False
    pos = copy(self.current_pos)
    info = {'prev_pos': copy(pos)}
    self.history_pos.extend(pos)
    # norm to 0~1
    self.history_pos[-2] /= 4.0
    self.history_pos[-1] /= 12.0
    if act == 0: # Up
      pos[0] -= 1
      if self._movable(pos):  # else stay in previous pos
        self.current_pos = pos
    elif act == 1:  # Down
      pos[0] += 1
      if self._movable(pos):
        self.current_pos = pos
    elif act == 2:  # Left
      pos[1] -= 1
      if self._movable(pos):
        self.current_pos = pos
    elif act == 3:  # Right
      pos[1] += 1
      if self._movable(pos):
        self.current_pos = pos
    elif act == 4:  # Still
      pass
    else:
      raise NotSupportedError("Invalid Action!")
    self.state[0], self.state[1] = self.current_pos[0]/4.0, self.current_pos[1]/12.0
    L1_dis_to_goal = np.abs(self.goal[0]-self.current_pos[0])+ np.abs(self.goal[1]-self.current_pos[1])
    self.state[2] = L1_dis_to_goal/10.0
    rew += (self._prev_dis - self.state[2])*1.0 # approaching reward
    self._prev_dis = self.state[2]
    next_pos = self.current_pos  # get new position after acting
    if self.map[next_pos[0], next_pos[1]] == 2:  # in water
      self.num_in_water += 1
      rew -= 2
    # next_obs = copy(self.history_pos)  # # shallow copy for list
    past_10_step = self.history_pos[-(2*self.history_step):]
    next_obs = copy(past_10_step)
    next_obs = np.zeros(self.history_step*2-len(past_10_step)).tolist() + next_obs
    next_obs += copy(self.state)  # append current pos and num of in water
    next_obs[-1] = self.num_in_water/4.0  # scale num of water by 4.0
    if self.cur_step >= self.MAX_STEP or (next_pos[0] == self.goal[0] and next_pos[1] == self.goal[1]):
      done = True
    self.cur_step += 1
    
    if next_pos[0] == self.goal[0] and next_pos[1] == self.goal[1]:
      rew += 10.0
    info['cur_pos'] = self.current_pos
    return next_obs, rew, done, info

  def _movable(self, pos):
    assert isinstance(pos[0], int) and isinstance(pos[1], int)
    if pos[0]<0 or pos[0]>4:
      return False
    if pos[1]<0 or pos[1]>12:
      return False
    if self.map[pos[0], pos[1]] == 1:
      return False
    else:
      return True

  # def render(self, display_time=0.1):
  #   pass
  #   row = self.map.shape[0]
  #   col = self.map.shape[1]
  #   unit = 20 * self._scale
  #   offset = unit
  #   for ii in range(row):
  #     tmp_row = self.map[ii, :]
  #     for jj in range(col):
  #       if tmp_row[jj] == 1:  # obstacle
  #         self._viewer.square((jj * unit + unit/2 + offset, ii * unit + unit/2 + offset), angle=0, edge=unit, color=(0,0,0))
  #       elif tmp_row[jj] == 2:  # water
  #         self._viewer.square((jj * unit + unit/2 + offset, ii * unit + unit/2 + offset), angle=45, edge=unit/(1.414), color=(0, 191, 255))
  #   # Draw a red circle for the agent
  #   self._viewer.circle((self.current_pos[1]*unit + unit/2 + offset, self.current_pos[0]*unit + unit/2 + offset), unit/(2.828), color=(255, 0, 0))

  #   # Draw a green circle for the goal
  #   self._viewer.circle((self.goal[1]*unit + unit/2 + offset, self.goal[0]*unit + unit/2 + offset), unit/4, color=(255, 192, 0))

  #   # Display the image for 0.1 seconds
  #   self._viewer.display(display_time)