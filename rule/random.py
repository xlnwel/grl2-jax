from core.elements.strategy import Strategy


class RandomStrategy(Strategy):
    def __init__(self, env, config, name='random'):
        super().__init__(
            name=name, 
            config=config, 
            env_stats=env.stats(), 
        )
        self.env = env
        self.n_envs = env.n_envs

    def __call__(self, *args, **kwargs):
        return [self.env.random_action() for _ in range(self.n_envs)]

def create_strategy(env, config, name='random'):
    strategy = RandomStrategy(env, config, name=name)
    return strategy
