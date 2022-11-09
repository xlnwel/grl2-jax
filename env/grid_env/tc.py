import collections
import numpy as np
from gym.spaces import Discrete

from core.typing import AttrDict
from tools.feature import one_hot
from tools.display import print_dict

DEFAULT_COLOURS = AttrDict({
    'red': np.array([255, 0, 0], np.uint8),
    'green': np.array([0, 255, 0], np.uint8),  
    'blue': np.array([0, 0, 255], np.uint8),  # Blue
})


class TwoColors:
    EID = 0

    def __init__(
        self,
        map_size=5, 
        step_penalty=-.04,
        max_episode_steps=100, 
        flip_steps=100000, 
        timeout_done=False, 
        reward_done=True,
        use_action_mask=False, 
        max_stay_still=None, 
        **kwargs
    ):
        self.map_size = map_size
        self.max_episode_steps = max_episode_steps
        self.step_penalty = step_penalty
        self.flip_steps = flip_steps
        self.timeout_done = timeout_done
        self.reward_done = reward_done
        self.use_action_mask = use_action_mask
        self.max_stay_still = max_stay_still

        obs_shape = (map_size * 6,)
        self.obs_shape = dict(
            obs=obs_shape, 
            global_state=obs_shape, 
        )
        self.obs_dtype = dict(
            obs=np.float32, 
            global_state=np.float32, 
        )
        if use_action_mask:
            self.obs_shape['action_mask'] = (4,)
            self.obs_dtype['action_mask'] = bool

        self._action_set = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], int)
        self.action_space = Discrete(4)
        self.action_shape = ()
        self.action_dim = 4
        self.action_dtype = np.int32
        self.is_action_discrete = True

        self.colors = AttrDict()
        self.color_keys = ['green', 'red', 'blue']

        self._steps = 0
        self.reward_colors = self.color_keys[1:]
        self.positive_color_idx = 0
        self.positive_color = self.reward_colors[0]

        self._prev_pos = None
        self._stay_still_times = 0
        self._max_stay_still = 0

        self._score = 0
        self._dense_score = 0
        self._episodic_score = 0
        self._epslen = 0
        self._step_scores = collections.deque(maxlen=1000)

        self.reset_map()

    def reset_map(self):
        """Resets the map to be empty as well as a custom reset set by subclasses"""
        points = []
        for _ in range(3):
            p = self._random_pos()
            while any([np.all(pp == p) for pp in points]):
                p = self._random_pos()
            points.append(p)
        self.colors.green = points[0]
        self.colors.blue = points[1]
        self.colors.red = points[2]
        self._prev_pos = self.colors.green
        self._stay_still_times = 0
        self._max_stay_still = 0

    def _random_pos(self):
        return np.random.randint(0, self.map_size, 2)
    
    def random_action(self):
        actions = np.random.randint(0, self.action_dim)
        return actions

    def get_screen(self):
        """Converts a map to an array of RGB values.
        Parameters
        ----------
        map: np.ndarray
            map to convert to colors
        color_map: dict
            mapping between array elements and desired colors
        Returns
        -------
        arr: np.ndarray
            3-dim numpy array consisting of color map
        """
        rgb = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        rgb[tuple(self.colors.red)] += DEFAULT_COLOURS.red
        rgb[tuple(self.colors.green)] += DEFAULT_COLOURS.green
        rgb[tuple(self.colors.blue)] += DEFAULT_COLOURS.blue
        return rgb

    def seed(self, seed):
        if seed is not None:
            np.random.seed(seed)

    def reset(self):
        self.reset_map()
        self.flip_reward_color()

        obs = self._get_obs()
        self._episodic_score = 0
        self._epslen = 0

        self._stay_still_times = 0

        return obs

    def step(self, action):
        self._steps += 1
        self.colors.green = self._move_green(action)

        reward = self._get_reward()

        self._score += reward
        self._dense_score += reward
        self._episodic_score += reward
        self._epslen += 1

        done = False
        if np.all(self.colors.green == self._prev_pos):
            self._stay_still_times += 1
        else:
            self._stay_still_times = 0
            self._prev_pos = self.colors.green
        if self.max_stay_still and self._stay_still_times > self.max_stay_still:
            done = True
            self.reset_map()
        self._max_stay_still = max(self._max_stay_still, self._stay_still_times)

        episodic_score = self._episodic_score
        epslen = self._epslen
        if done or reward != self.step_penalty:
            self._step_scores.append(self._episodic_score / self._epslen)
            self._episodic_score = 0
            self._epslen = 0
            if not done:
                done = self.reward_done
        step_score = self._dense_score / self._steps
        recent_step_score = np.mean(self._step_scores) \
            if len(self._step_scores) else step_score
        game_over = done if self.reward_done \
            else self._steps >= self.max_episode_steps

        obs = self._get_obs()

        info = {
            'score': self._score, 
            'dense_score': self._dense_score, 
            'episodic_score': episodic_score, 
            'recent_step_score': recent_step_score, 
            'step_score': step_score, 
            'epslen': epslen, 
            'game_over': game_over,
            'max_stay_still': self._max_stay_still, 
        }

        return obs, reward, done, info

    def _get_obs(self):
        obs = []
        for c in self.color_keys:
            obs = sum([one_hot(p, self.map_size) for p in self.colors[c]], obs)
        obs = {
            'obs': np.asarray(obs, np.float32), 
            'global_state': np.asarray(obs, np.float32),
        }
        if self.use_action_mask:
            obs['action_mask'] = self._get_action_mask()
        return obs

    def _get_action_mask(self):
        action_mask = np.ones(4, bool)
        for a in range(4):
            green = self._move_green(a)
            if np.all(green == self.colors.green):
                action_mask[a] = False
        return action_mask

    def _move_green(self, action):
        action = self._action_set[action]
        # print('before move', self.colors.green)
        green = self.colors.green + action
        green = np.clip(green, 0, self.map_size-1)
        # print('after move', self.colors.green)
        return green

    def _get_reward(self):
        if np.all(self.colors.green == self.colors.red):
            reward = 1 if self.positive_color == 'red' else -1
            # print('reward', reward)
            # print('hit red at', self.colors.green, self.colors.red)
            # if not self.reward_done:
            self.colors.red = self._random_pos()
            while any([np.all(self.colors.red == p) for p in [self.colors.green, self.colors.blue]]):
                self.colors.red = self._random_pos()
            # print('resprawn red at', self.colors.red)
        elif np.all(self.colors.green == self.colors.blue):
            reward = 1 if self.positive_color == 'blue' else -1
            # print('reward', reward)
            # print('hit blue at', self.colors.green, self.colors.blue)
            # if not self.reward_done:
            self.colors.blue = self._random_pos()
            while any([np.all(self.colors.blue == p) for p in [self.colors.green, self.colors.red]]):
                self.colors.blue = self._random_pos()
            # print('resprawn blue at', self.colors.blue)
        else:
            reward = self.step_penalty
        return reward

    def flip_reward_color(self):
        if self._steps >= self.flip_steps:
            self.positive_color_idx = (self.positive_color_idx+1) % len(self.reward_colors)
            self.positive_color = self.reward_colors[self.positive_color_idx]
            self._step_scores.clear()
            self._dense_score = 0
            self._steps = 0
            print('positive color', self.positive_color)


if __name__ == '__main__':
    env = TwoColors(5, step_penalty=0, max_episode_steps=50, flip_steps=10)
    obs = env.reset()
    for i in range(100):
        a = env.random_action()
        prev_obs = obs
        obs, reward, done, info = env.step(a)
        if reward != 0:
            print(prev_obs['obs'].reshape(-1, 5).argmax(-1), obs['obs'].reshape(-1, 5).argmax(-1), sep='\n')
