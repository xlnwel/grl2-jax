from core.elements.actor import Actor


def create_actor(config, model, name='zero'):
    return Actor(config=config, model=model, name=name)
