from replay.uniform import UniformReplay


def create_buffer(config, model, env_stats, **kwargs):
    return UniformReplay(
        config=config, 
        env_stats=env_stats, 
        model=model, 
        **kwargs
    )
