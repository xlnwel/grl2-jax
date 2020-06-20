import numpy as np
import os
os.environ.setdefault('PATH', '')
from collections import deque
import gym
from gym.spaces.box import Box
import cv2
cv2.ocl.setUseOpenCL(False)

from utility.display import pwc
from env import baselines as B


def make_atari_env(config):
    assert 'atari' in config['name']
    if config.setdefault('wrapper', 'baselines') == 'baselines':
        name = config['name'].split('_', 1)[-1]
        name = name[0].capitalize() + name[1:]
        version = 0 if config.setdefault('sticky_actions', True) else 4
        name = f'{name}NoFrameskip-v{version}'
        env = B.make_atari(name, config.setdefault('frame_skip', 4))
        env = B.wrap_deepmind(env, 
            episode_life=config.setdefault('life_done', False), 
            clip_reward=config.setdefault('clip_reward', False),
            frame_stack=config.setdefault('frame_stack', 4),
            np_obs=config.setdefault('np_obs', False))
    else:
        env = Atari(**config)
    config.setdefault('max_episode_steps', 108000)    # 30min

    return env

class Atari:
    """A class implementing image preprocessing for Atari 2600 agents.
    Code is originally from Dopamine, adapted to more general setting

    Specifically, this provides the following subset from the JAIR paper
    (Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):

    * Frame skipping (defaults to 4).
    * Terminal signal when a life is lost (off by default).
    * Grayscale and max-pooling of the last two frames.
    * Downsample the screen to a square image (defaults to 84x84).

    More generally, this class follows the preprocessing guidelines set down in
    Machado et al. (2018), "Revisiting the Arcade Learning Environment:
    Evaluation Protocols and Open Problems for General Agents".
    """

    def __init__(self, name, *, frame_skip=4, life_done=False,
                image_size=(84, 84), frame_stack=1, noop=30, 
                sticky_actions=True, gray_scale=True, 
                clip_reward=False, np_obs=False, **kwargs):
        version = 0 if sticky_actions else 4
        name = name.split('_', 1)[-1]
        name = name[0].capitalize() + name[1:]
        name = f'{name}NoFrameskip-v{version}'
        env = gym.make(name)
        # Strip out the TimeLimit wrapper from Gym, which caps us at 100k frames. 
        # We handle this time limit internally instead, which lets us cap at 108k 
        # frames (30 minutes). The TimeLimit wrapper also plays poorly with 
        # saving and restoring states.
        self.env = env.env
        self.life_done = life_done
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.gray_scale = gray_scale
        self.noop = noop
        self.image_size = (image_size, image_size) \
            if isinstance(image_size, int) else tuple(image_size)
        self.clip_reward = clip_reward
        self.np_obs = np_obs

        assert self.frame_skip > 0, \
            f'Frame skip should be strictly positive, got {self.frame_skip}'
        assert self.frame_stack > 0, \
            f'Frame stack should be strictly positive, got {self.frame_stack}'
        assert np.all([s > 0 for s in self.image_size]), \
            f'Screen size should be strictly positive, got {image_size}'

        obs_shape = self.env.observation_space.shape
        # Stores temporary observations used for pooling over two successive
        # frames.
        shape = obs_shape[:2]
        if not gray_scale:
            shape += (3,)
        self._buffer = [np.empty(shape, dtype=np.uint8) for _ in range(2)]

        self.lives = 0  # Will need to be set by reset().
        self._game_over = True
        if self.frame_stack > 1:
            # Stores LazyFrames for memory efficiency
            self._frames = deque([], maxlen=self.frame_stack)

    @property
    def observation_space(self):
        # Return the observation space adjusted to match the shape of the processed
        # observations.
        c = 1 if self.gray_scale else 3
        c *= self.frame_stack
        shape = self.image_size + (c, )
        return Box(low=0, high=255, shape=shape,
                dtype=np.uint8)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def reward_range(self):
        return self.env.reward_range

    @property
    def metadata(self):
        return self.env.metadata

    def seed(self, seed=0):
        self.env.seed(seed)

    def close(self):
        return self.env.close()

    def get_screen(self):
        return self.env.ale.getScreenRGB2()

    def game_over(self):
        return self._game_over

    def set_game_over(self):
        self._game_over = True

    def reset(self, **kwargs):
        if self._game_over:
            self.env.reset(**kwargs)
            if 'FIRE' in self.env.get_action_meanings():
                action = self.env.get_action_meanings().index('FIRE')
                for _ in range(self.frame_skip):
                    self.env.step(action)
            noop = np.random.randint(1, self.noop + 1)
            for _ in range(noop):
                d = self.env.step(0)[2]
                if d:
                    self.env.reset(**kwargs)
        else:
            if 'FIRE' in self.env.get_action_meanings():
                action = self.env.get_action_meanings().index('FIRE')
            else:
                action = 0
            self.step(action)
        
        self.lives = self.env.ale.lives()
        self._get_screen(self._buffer[0])
        self._buffer[1].fill(0)
        obs = self._pool_and_resize()
        if self.frame_stack > 1:
            for _ in range(self.frame_stack):
                self._frames.append(obs)
            obs = self._get_obs()

        self._game_over = False
        return obs

    def render(self, mode):
        """Renders the current screen, before preprocessing.

        This calls the Gym API's render() method.

        Args:
            mode: Mode argument for the environment's render() method.
                Valid values (str) are:
                'rgb_array': returns the raw ALE image.
                'human': renders to display via the Gym renderer.

        Returns:
            if mode='rgb_array': numpy array, the most recent screen.
            if mode='human': bool, whether the rendering was successful.
        """
        return self.env.render(mode)

    def step(self, action):
        total_reward = 0.

        for step in range(1, self.frame_skip+1):
            # We bypass the Gym observation altogether and directly fetch
            # the image from the ALE. This is a little faster.
            _, reward, done, info = self.env.step(action)
            if self.clip_reward:
                reward = np.clip(reward, -1, 1)
            total_reward += reward

            if self.life_done:
                new_lives = self.env.ale.lives()
                is_terminal = done or new_lives < self.lives
                self.lives = new_lives
            else:
                is_terminal = done

            if is_terminal:
                break
            elif step >= self.frame_skip - 1:
                i = step - (self.frame_skip - 1)
                self._get_screen(self._buffer[i])

        # Pool the last two observations.
        obs = self._pool_and_resize()
        if self.frame_stack > 1:
            self._frames.append(obs)
            obs = self._get_obs()

        self._game_over = done
        info['frame_skip'] = step
        return obs, total_reward, is_terminal, info

    def _pool_and_resize(self):
        """Transforms two frames into a Nature DQN observation.

        For efficiency, the transformation is done in-place in self._buffer.

        Returns:
            transformed_screen: numpy array, pooled, resized image.
        """
        # Pool if there are enough screens to do so.
        if self.frame_skip > 1:
            np.maximum(self._buffer[0], self._buffer[1],
                    out=self._buffer[0])

        img = cv2.resize(
            self._buffer[0], self.image_size, interpolation=cv2.INTER_AREA)
        img = np.asarray(img, dtype=np.uint8)
        return np.expand_dims(img, axis=2) if self.gray_scale else img

    def _get_screen(self, output):
        if self.gray_scale:
            self.env.ale.getScreenGrayscale(output)
        else:
            self.env.ale.getScreenRGB2(output)

    def _get_obs(self):
        assert len(self._frames) == self.frame_stack, f'{len(self._frames)} vs {self.frame_stack}'
        return np.concatenate(self._frames, axis=-1) \
            if self.np_obs else B.LazyFrames(list(self._frames))


if __name__ == '__main__':
    config = dict(
        name='breakout',
        precision=32,
        life_done=True,
        sticky_actions=True,
        seed=0
    )

    import numpy as np
    from env.baselines import make_atari, wrap_deepmind
    np.random.seed(0)
    name = config['name']
    version = 0
    name = f'{name.title()}NoFrameskip-v{version}'
    print(name)
    env1 = make_atari(name)
    env1 = wrap_deepmind(env1, episode_life=config['life_done'], frame_stack=True)
    np.random.seed(0)
    env2 = Atari(**config)
    env1.seed(0)
    env2.seed(0)
    np.random.seed(0)
    o1 = env1.reset()
    np.random.seed(0)
    o2 = env2.reset()
    o1 = env1.ale.getScreenRGB2()
    o2 = env2.get_screen()
    np.testing.assert_allclose(np.array(o1), np.array(o2))
    for i in range(1,1000):
        np.random.seed(0)
        o1, r1, d1, _ = env1.step(2)
        o1 = env1.ale.getScreenRGB2()
        np.random.seed(0)
        print(i, env1.lives, env2.lives)
        o2, r2, d2, _ = env2.step(2)
        o2 = env2.env.ale.getScreenRGB2()
        np.testing.assert_allclose(np.array(o1), np.array(o2))
        np.testing.assert_allclose(r1, r2)
        np.testing.assert_allclose(d1, d2)
        if d1:
            env1.reset()
            env2.reset()