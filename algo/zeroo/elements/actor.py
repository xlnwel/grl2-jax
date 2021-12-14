from algo.zero.elements.actor import PPOActor


class OPPOActor(PPOActor):
    def _post_init(self):
        super()._post_init()
        self.config.algorithm = 'zeroo'


def create_actor(config, model, name='ppo'):
    return OPPOActor(config=config, model=model, name=name)
