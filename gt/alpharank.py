from typing import List
import numpy as np
from scipy.linalg import eig


class AlphaRank:
    def __init__(
        self, 
        alpha, 
        m=5, 
        epsilon=1e-5
    ):
        self.alpha = alpha
        self.m = m
        self.epsilon = epsilon

    """ Rank """
    def compute_rank(
        self, 
        payoffs: List[np.ndarray], 
        is_single_population=False, 
        return_mass=False
    ):
        """ Return ranks and corresponding masses """
        if is_single_population:
            return self.compute_rank_sp(payoffs, return_mass)
        else:
            return self.compute_rank_mp(payoffs, return_mass)

    def compute_rank_sp(
        self, 
        payoffs: List[np.ndarray], 
        return_mass=False
    ):
        pi = self.compute_stationary_distribution(payoffs, True)
        rank = np.argsort(pi)[::-1]
        if return_mass:
            return rank, pi[rank]
        else:
            return rank

    def compute_rank_mp(
        self, 
        payoffs: List[np.ndarray], 
        return_mass=False
    ):
        pi = self.compute_stationary_distribution(payoffs, False)

        ns = payoffs[0].shape
        strategy_masses = [np.zeros(n) for n in ns]
        for si, p in enumerate(pi):
            sp = self._idx2sp(si, ns)
            for a, s in zip(strategy_masses, sp):
                a[s] += p

        agent_ranks = [np.argsort(s)[::-1] for s in strategy_masses]
        if return_mass:
            return agent_ranks, [s[r] for s, r in zip(strategy_masses, agent_ranks)]
        else:
            return agent_ranks

    """ Stationary Distribution """
    def compute_stationary_distribution(
        self, 
        payoffs: List[np.ndarray], 
        is_single_population=False
    ):
        # print('payoff', *payoffs, sep='\n')
        transition = self.compute_transition_matrix(payoffs, is_single_population)
        # print('transition', transition, sep='\n')
        eig_vals, eig_vecs = eig(transition, left=True, right=False)
        # print('eigen values', np.real(eig_vals))
        mask = np.isclose(1-eig_vals, 1e-10)
        if np.sum(mask) != 1:
            raise ValueError(
                f'Expected 1 stationary distribution, but found {np.sum(mask)}')
        # print('eigen vectors', eig_vecs)
        eig_vec = eig_vecs[np.arange(eig_vals.size), mask]
        eig_vec = np.real(eig_vec)
        # print('eigen vector', eig_vec)
        pi = eig_vec / np.sum(eig_vec)
        return pi
    
    """ Transition Matrix """
    def compute_transition_matrix(
        self, 
        payoffs: List[np.ndarray], 
        is_single_population=False
    ):
        if is_single_population:
            assert len(payoffs) == 1, len(payoffs)
            return self._compute_transition_matrix_sp(payoffs)
        else:
            assert len(payoffs) == payoffs[0].ndim, (len(payoffs), payoffs[0].ndim)
            return self._compute_transition_matrix_mp(payoffs)
        
    def _compute_transition_matrix_sp(
        self, 
        payoffs: List[np.ndarray]
    ):
        assert len(payoffs) == 1, len(payoffs)
        for p in payoffs:
            assert p.ndim == 2, p.shape
            np.testing.assert_allclose(p.T, -p)
        payoff = payoffs[0]
        n_strategies = payoff.shape[0]
        eta = 1 / (n_strategies - 1)
        transition_matrix = np.zeros_like(payoff, dtype=np.float64)

        for s1 in range(n_strategies):
            for s2 in range(n_strategies):
                if s1 != s2:
                    transition_matrix[s1, s2] = self._compute_transition_probability(
                        payoff[s2, s1], eta)
            pair_transition = np.sum(transition_matrix[s1])
            assert np.sum(transition_matrix[s1]) <= 1, (s1, transition_matrix[s1])
            transition_matrix[s1, s1] = 1 - pair_transition
        
        return transition_matrix

    def _compute_transition_matrix_mp(
        self, 
        payoffs: List[np.ndarray]
    ):
        def get_different_dim(s1, s2, ns):
            diff_dim = None
            sp1, sp2 = [], []
            for k, _ in enumerate(ns):
                n = np.prod(ns[k+1:], dtype=np.int32)
                i = 0 if s1 < n else s1 // n
                j = 0 if s2 < n else s2 // n
                s1 -= i * n
                s2 -= j * n
                sp1.append(i)
                sp2.append(j)
                if i != j:
                    if s1 == s2:
                        diff_dim = k
                    else:
                        return None, None, None

            return diff_dim, tuple(sp1), tuple(sp2)

        n_agent_strategies = payoffs[0].shape
        for p in payoffs:
            assert n_agent_strategies == p.shape
        
        eta = 1 / np.sum([n-1 for n in n_agent_strategies])

        n_strategy_profiles = np.prod(n_agent_strategies)
        transition_matrix = np.zeros((n_strategy_profiles, n_strategy_profiles), dtype=np.float64)

        for s1 in range(n_strategy_profiles):
            for s2 in range(n_strategy_profiles):
                k, sp1, sp2 = get_different_dim(s1, s2, n_agent_strategies)
                if k is not None:
                    transition_matrix[s1, s2] = self._compute_transition_probability(
                        payoffs[k][sp2] - payoffs[k][sp1], eta)
            pair_transition = np.sum(transition_matrix[s1])
            assert np.sum(transition_matrix[s1]) <= 1, (s1, transition_matrix[s1])
            transition_matrix[s1, s1] = 1 - pair_transition

        return transition_matrix
    
    def _compute_transition_probability(self, delta_f, eta):
        fix_prob = self._compute_fixation_probability(delta_f)
        fix_prob = min(fix_prob + self.epsilon, 1)
        trans_prob = eta * fix_prob
        return trans_prob

    """ Fixation Matrix """
    def compute_fixation_matrix(
        self, 
        payoffs: List[np.ndarray], 
        is_single_population=False
    ):
        if is_single_population:
            assert len(payoffs) == 1, len(payoffs)
            return self._compute_fixation_matrix_sp(payoffs)
        else:
            assert len(payoffs) == payoffs[0].ndim, (len(payoffs), payoffs[0].ndim)
            return self._compute_fixation_matrix_mp(payoffs)
        
    def _compute_fixation_matrix_sp(
        self, 
        payoffs: List[np.ndarray]
    ):
        assert len(payoffs) == 1, len(payoffs)
        for p in payoffs:
            assert p.ndim == 2, p.shape
            np.testing.assert_allclose(p.T, -p)
        payoff = payoffs[0]
        n_strategies = payoff.shape[0]
        fixation_matrix = np.zeros_like(payoff)

        for s1 in range(n_strategies):
            for s2 in range(n_strategies):
                if s1 != s2:
                    fixation_matrix[s1, s2] = self._compute_fixation_probability(
                        payoff[s2, s1])
            fixation_matrix[s1, s1] = 1 / self.m

        return fixation_matrix

    def _compute_fixation_matrix_mp(
        self, 
        payoffs: List[np.ndarray]
    ):
        def get_different_dim(s1, s2, ns):
            diff_dim = None
            sp1, sp2 = [], []
            for k, _ in enumerate(ns):
                n = np.prod(ns[k+1:], dtype=np.int32)
                i = 0 if s1 < n else s1 // n
                j = 0 if s2 < n else s2 // n
                s1 -= i * n
                s2 -= j * n
                sp1.append(i)
                sp2.append(j)
                if i != j:
                    if s1 == s2:
                        diff_dim = k
                    else:
                        return None, None, None

            return diff_dim, tuple(sp1), tuple(sp2)

        n_agent_strategies = payoffs[0].shape
        for p in payoffs:
            assert n_agent_strategies == p.shape

        n_strategy_profiles = np.prod(n_agent_strategies)
        fixation_matrix = np.zeros((n_strategy_profiles, n_strategy_profiles))

        for s1 in range(n_strategy_profiles):
            for s2 in range(n_strategy_profiles):
                k, sp1, sp2 = get_different_dim(s1, s2, n_agent_strategies)
                if k is not None:
                    fixation_matrix[s1, s2] = self._compute_fixation_probability(
                        payoffs[k][sp2] - payoffs[k][sp1])
            fixation_matrix[s1, s1] = 1 / self.m

        return fixation_matrix

    def _compute_fixation_probability(self, delta_f):
        if np.isclose(delta_f, 0):
            return 1 / self.m
        e1 = np.exp(-self.alpha * delta_f)
        e2 = e1**self.m
        if np.isinf(e2):
            return 0
        rho = (1 - e1) / (1 - e2)
        return rho

    def _idx2sp(self, si, ns):
        """ Convert strategy profile index (si) to a strategy profile """
        sp = []
        for k, _ in enumerate(ns):
            n = np.prod(ns[k+1:], dtype=np.int32)
            i = si // n
            si -= i * n
            sp.append(i)
        return sp


if __name__ == '__main__':
    alpha_rank = AlphaRank(1000, 5, 0)

    # biased rock-papaer-scissors
    payoffs = [
        np.array([
            [0, -.5, 1],
            [.5, 0, -.1],
            [-1, .1, 0],
        ])
    ]
    pi = alpha_rank.compute_stationary_distribution(payoffs, True)
    np.testing.assert_allclose(pi, 1/3)


    phi = 10
    eps = .1
    payoffs = [
        np.array([
            [0, -phi, 1, phi, -eps], 
            [phi, 0, -phi**2, 1, -eps], 
            [-1, phi**2, 0, -phi, -eps], 
            [-phi, -1, phi, 0, -eps], 
            [eps, eps, eps, eps, 0]
        ])
    ]
    pi = alpha_rank.compute_stationary_distribution(payoffs, True)
    rank = alpha_rank.compute_rank(payoffs, True)
    assert rank[0] == 4, rank
    payoffs.append(payoffs[0].T.copy())
    pi = alpha_rank.compute_stationary_distribution(payoffs, False)

    payoffs = [
        np.array([
            [0, 4.6, -4.6, -4.6], 
            [-4.6, 0, 4.6, 4.6], 
            [4.6, -4.6, 0, 0], 
            [4.6, -4.6, 0, 0], 
        ])
    ]

    pi = alpha_rank.compute_stationary_distribution(payoffs, True)
    np.testing.assert_allclose(pi, [.2, .4, .2, .2], atol=1e-2)

    payoffs.append(payoffs[0].T.copy())
    pi = alpha_rank.compute_stationary_distribution(payoffs)

    payoffs = [
        np.array([
            [0, 4.6, -4.6], 
            [-4.6, 0, 4.6], 
            [4.6, -4.6, 0], 
        ])
    ]
    pi = alpha_rank.compute_stationary_distribution(payoffs, True)
    np.testing.assert_allclose(pi, 1/3)

    payoffs.append(payoffs[0].T.copy())
    pi = alpha_rank.compute_stationary_distribution(payoffs)
    np.testing.assert_allclose(pi, 1/9)

    payoffs = [
        np.array([
            [3, 0],
            [0, 2]
        ]),
        np.array([
            [2, 0],
            [0, 3]
        ])
    ]
    fix = alpha_rank.compute_fixation_matrix(payoffs)
    pi = alpha_rank.compute_stationary_distribution(payoffs)
    np.testing.assert_allclose(pi, [.5, 0, 0, .5], atol=1e-2)

    n = 5
    payoff = np.random.normal(size=(n, n))
    payoff = payoff - payoff.T
    payoff = np.array([
        [ 0.        ,  2.165003  ,  1.44582028, -0.26998757,  0.45503604],
        [-2.165003  ,  0.        , -1.51042755,  0.96713084, -0.45672137],
        [-1.44582028,  1.51042755,  0.        , -0.18188951,  1.92780254],
        [ 0.26998757, -0.96713084,  0.18188951,  0.        ,  1.78196242],
        [-0.45503604,  0.45672137, -1.92780254, -1.78196242,  0.        ],
    ])
    payoffs = [payoff]
    flow = np.sum(payoffs[0] > 0, -1)
    print('payoff', *payoff, sep='\n')
    print('flow', flow)
    print('fixation matrix', alpha_rank.compute_fixation_matrix(payoffs, True), sep='\n')
    print('transition matrix', alpha_rank.compute_transition_matrix(payoffs, True), sep='\n')
    rank, mass = alpha_rank.compute_rank(payoffs, True, True)
    p2 = flow[rank]
    print('rank', rank)
    print('argsort', np.argsort(flow)[::-1])
    np.testing.assert_equal(p2, np.sort(flow)[::-1])
    # np.testing.assert_allclose(rank, np.argsort(p)[::-1])
