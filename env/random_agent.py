from env.typing import EnvOutput


class RandomAgent:
    def __init__(self, env):
        self.env = env
    
    """ Call """
    def __call__(
        self, 
        env_output: EnvOutput, 
        evaluation: bool=False,
        return_eval_stats: bool=False
    ):
        action = self.env.random_action()

        if 'eid' in env_output.obs:
            action = action['eid']

        return action, {}
