import numpy as np

from gt.utils import compute_opponent_distribution


class FSP:
    def __init__(self, **kwargs):
        pass

    def __call__(
        self, 
        aid: int, 
        model_payoff: np.ndarray, 
        n_agents: int, 
        prioritize_unmet: bool=True
    ):
        """ Fictitious Self-Play """
        assert len(model_payoff.shape) == n_agents - 1, (model_payoff.shape, n_agents)
        payoffs, dists = compute_opponent_distribution(
            aid, 
            model_payoff, 
            n_agents, 
            prioritize_unmet=prioritize_unmet, 
            reweight_kwargs={
                'p': 0, 
                'type': 'uniform'
            }
        )

        return payoffs, dists
