import numpy as np

from gt.utils import compute_opponent_distribution


class PFSP:
    def __init__(
        self, 
        p, 
        threshold=.1, 
        **kwargs
    ):
        self._p = p
        self._threshold = threshold

    def __call__(
        self, 
        aid: int, 
        model_payoff: np.ndarray, 
        n_agents: int, 
    ):
        """ Prioritized Fictitous Self-Play """
        assert len(model_payoff.shape) == n_agents - 1, (model_payoff.shape, n_agents)
        payoffs, dists = compute_opponent_distribution(
            aid, 
            model_payoff, 
            n_agents, 
            self._threshold, 
            {'p': self._p, 'type': 'reverse_poly'}
        )

        return payoffs, dists
