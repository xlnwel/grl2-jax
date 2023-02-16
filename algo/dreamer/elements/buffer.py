from replay.eps import EpisodicReplay


def create_buffer(config, model, env_stats, **kwargs):
    return EpisodicReplay(
        config=config, 
        env_stats=env_stats, 
        model=model, 
        **kwargs
    )