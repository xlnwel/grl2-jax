from core.agent import Agent


def collect(buffer, env, env_step, reset, next_obs, **kwargs):
    kwargs['reset'] = reset
    buffer.add(**kwargs)


def create_agent(**kwargs):
    return Agent(**kwargs)
