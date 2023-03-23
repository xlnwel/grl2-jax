from core.elements.agent import Agent as AgentBase


class Agent(AgentBase):
    def _post_init(self):
        self.trainer.sync_lookahead_params()


def create_agent(**kwargs):
    return Agent(**kwargs)
