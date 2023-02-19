from replay import buffer_registry

def create_buffer(config, model, env_stats, **kwargs):
    BufferCls = buffer_registry.get(config.type)
    return BufferCls(
            config=config, 
        env_stats=env_stats, 
        model=model, 
        **kwargs
    )
