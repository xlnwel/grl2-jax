import numpy as np

from gt.alpharank import AlphaRank
from gt.rm import RegretMatching
from gt.utils import compute_utility


class TestClass:
    def test_compute_utility(self):
        n = np.random.randint(3, 10)
        shape = tuple([np.random.randint(3, 10) for _ in range(n)])
        payoff = np.random.normal(size=shape)
        strategy = [np.random.normal(size=(i,)) for i in shape]
        x = compute_utility(payoff, strategy)
        y = compute_utility(payoff, strategy, True)
        np.testing.assert_allclose(x, y)
        z = compute_utility(payoff, strategy[:2], True)
        z = compute_utility(z, strategy[2:])
        np.testing.assert_allclose(x, z)
        assert isinstance(z, float)

    def test_regret_matching(self):
        m, n = 2, 2
        A = np.array([[2, -1], [-1, 1]])
        B = -np.array([[2, -1], [-1, 1]])
        payoffs = [A, -B]

        rm = RegretMatching(n, m)
        ans = rm.compute(payoffs, n, m, 10000)
        np.testing.assert_approx_equal(ans[0], np.array([0.4, 0.6]), 1)
        np.testing.assert_approx_equal(ans[1], np.array([0.4, 0.6]), 1)

        payoffs = [np.array([[5, 0], [0, 3]]), np.array([[3, 0], [0, 5]])]
        rm = RegretMatching(n, m)
        ans = rm.compute(payoffs, n, m, 10000)
        np.testing.assert_approx_equal(ans[0], np.array([0.625, 0.375]))
        
    def test_alpha_rank(self):
        alpha_rank = AlphaRank(100, 5)

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
        np.testing.assert_allclose(pi, [.2, .4, .2, .2], atol=1e-4)

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
        pi = alpha_rank.compute_stationary_distribution(payoffs)
        np.testing.assert_allclose(pi, [.5, 0, 0, .5], atol=1e-4)

        n = 5
        payoff = np.random.normal(size=(n, n))
        payoff = payoff - payoff.T
        payoffs = [payoff]
        flow = np.sum(payoffs[0] > 0, -1)
        rank, mass = alpha_rank.compute_rank(payoffs, True, True)
        p2 = flow[rank]
        np.testing.assert_equal(p2, np.sort(flow)[::-1])
