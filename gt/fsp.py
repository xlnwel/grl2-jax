import numpy as np

from gt import utils

class FSP:
    def __init__(self, **kwargs):
        pass

    def __call__(
        self, 
        aid: int, 
        model_payoff: np.ndarray, 
        n_agents: int, 
        prioritize_unmet: bool, 
        filter_recent: bool
    ):
        """ Fictitious Self-Play """
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
                'p': 0, 
                'type': 'uniform'
            }, 
            filter_recent=filter_recent
        )
        dists = utils.compute_opponent_distribution(weights)
        return payoffs, dists
