from core.elements.agent import create_agent


def collect(buffer, env, env_step, reset, next_obs, **kwargs):
    kwargs['reset'] = reset
    buffer.add(**kwargs)
