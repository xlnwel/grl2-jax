import numpy as np

from gt import utils


class PFSP:
    def __init__(
        self, 
        p, 
        threshold=0, 
        **kwargs
    ):
        self._p = p
        self._threshold = threshold

    def __call__(
        self, 
        aid: int, 
        model_payoff: np.ndarray, 
        n_agents: int, 
        prioritize_unmet: bool, 
        filter_recent: bool
    ):
        """ Prioritized Fictitous Self-Play """
        assert len(model_payoff.shape) == n_agents - 1, (model_payoff.shape, n_agents)
        payoffs = utils.get_opponent_payoffs(
            aid, 
            model_payoff, 
            n_agents, 
        )
        weights = utils.compute_opponent_weights(
            payoffs, 
            prioritize_unmet=prioritize_unmet, 
            reweight_kwargs={
                'p': self._p, 
                'type': 'poly', 
                'threshold': self._threshold
            }, 
            filter_recent=filter_recent
        )
        dists = utils.compute_opponent_distribution(weights)
        return payoffs, dists
