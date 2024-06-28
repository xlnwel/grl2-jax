import gym


class FlyControl(gym.Wrapper):
  def __init__(self, env: gym.Env):
    super().__init__(env)
    self.goal = None
  