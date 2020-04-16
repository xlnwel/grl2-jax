import numpy as np
import os
os.environ.setdefault('PATH', '')
from collections import deque
import gym
from gym.spaces.box import Box
import cv2
cv2.ocl.setUseOpenCL(False)

from utility.display import pwc
from env.baselines import make_atari, wrap_deepmind

def make_atari_env(config):
    # name = config['name']
    # version = 0
    # name = f'{name.title()}NoFrameskip-v{version}'
    # env = make_atari(name)
    # env = wrap_deepmind(env, frame_stack=True)
    env = Atari(**config)
    return env


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames

    def _force(self):
        out = np.concatenate(self._frames, axis=-1)
        return out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[-1]

    def frame(self, i):
        return self._force()[..., i]

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
                image_size=(84, 84), frame_stack=4, noop=30, 
                sticky_actions=True, gray_scale=True, 
                **kwargs):
        version = 0 if 'sticky_actions' else 4
        name = f'{name.title()}NoFrameskip-v{version}'
        env = gym.make(name)
        print(f'Environment name: {name}')
        # Strip out the TimeLimit wrapper from Gym, which caps us at 100k frames. We
        # handle this time limit internally instead, which lets us cap at 108k frames
        # (30 minutes). The TimeLimit wrapper also plays poorly with saving and
        # restoring states.
        self.env = env.env
        self.life_done = life_done
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.gray_scale = gray_scale
        self.noop = noop
        self.image_size = (image_size, image_size) \
            if isinstance(image_size, int) else tuple(image_size)

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

        self._game_over = True
        self.lives = 0  # Will need to be set by reset().
        # Stores LazyFrames for memory efficiency
        self._frames = deque([], maxlen=self.frame_stack)

    def get_screen(self):
        return self.env.ale.getScreenRGB2()

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

    def reset(self, **kwargs):
        if self._game_over:
            self.env.reset(**kwargs)
            if 'FIRE' in self.env.unwrapped.get_action_meanings():
                # Fire when reset
                done = self.env.step(1)[2]
                if done:
                    self.env.reset(**kwargs)
                done = self.env.step(2)[2]
                if done:
                    self.env.reset(**kwargs)
            noop = np.random.randint(1, self.noop + 1)
            for _ in range(noop):
                d = self.env.step(0)[2]
                if d:
                    self.env.reset(**kwargs)
        else:
            self.env.step(0)

        self.lives = self.env.ale.lives()
        self._get_screen(self._buffer[0])
        self._buffer[1].fill(0)
        obs = self._pool_and_resize()
        for _ in range(self.frame_stack):
            self._frames.append(obs)
        return self._get_obs()

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
        accumulated_reward = 0.

        for step in range(self.frame_skip):
            # We bypass the Gym observation altogether and directly fetch
            # the image from the ALE. This is a little faster.
            _, reward, done, info = self.env.step(action)
            accumulated_reward += reward

            if self.life_done:
                new_lives = self.env.ale.lives()
                is_terminal = done or new_lives < self.lives
                self.lives = new_lives
            else:
                is_terminal = done

            if is_terminal:
                break
            elif step >= self.frame_skip - 2:
                i = step - (self.frame_skip - 2)
                self._get_screen(self._buffer[i])

        # Pool the last two observations.
        obs = self._pool_and_resize()
        self._frames.append(obs)
        obs = self._get_obs()
        self._game_over = done
        info['n_ar'] = self.frame_skip
        return obs, accumulated_reward, is_terminal, info

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
        assert len(self._frames) == self.frame_stack
        return LazyFrames(list(self._frames))


if __name__ == '__main__':
    config = dict(
        name='breakout', 
        max_episode_steps=27000,
        life_done=True
    )
    env = make_atari_env(config)
    env.seed(0)
    o = env.reset()
    d = np.zeros(len(o))
    for k in range(0, 10):
        o = np.array(o)
        a = env.action_space.sample()
        no, r, d, i = env.step(a)
        print(k, r, d, env.lives)
        if d:
            np.testing.assert_equal(np.array(o)[...,1:], np.array(no)[...,:-1])
            print(k, env.lives, d, env._game_over)
            env.reset()
        o = no