from core.elements.agent import Agent as AgentBase


class Agent(AgentBase):
    def _post_init(self):
        self.trainer.sync_lookahead_params()

    def lookahead_train(self, **kwargs):
        stats = self.strategy.lookahead_train(**kwargs)
        if stats is not None:
            self.monitor.store(**stats)


def create_agent(**kwargs):
    return Agent(**kwargs)
