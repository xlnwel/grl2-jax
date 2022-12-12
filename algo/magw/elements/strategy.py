from core.elements.strategy import Strategy, create_strategy


def choose_elites(self, idx=None):
    self.model.choose_elites(idx)

Strategy.choose_elites = choose_elites
