import numpy as np

from gt.utils import compute_utility



class RegretMatching:
    def __init__(self, n_agents, n_strategies):
        self.reset(n_agents, n_strategies)

    def reset(self, n_agents, n_strategies):
        self.regrets = [
            np.zeros(n_strategies, np.float32) for _ in range(n_agents)]
        self.meta_strategies = [
            np.ones(n_strategies, np.float32) / n_strategies for _ in range(n_agents)]
        self.average_meta_strategies = [
            np.ones(n_strategies) / n_strategies for _ in range(n_agents)]
        self.n_meta_strategy_updates = [0 for _ in range(n_agents)]

    def update_regrets(self, payoff, aid, sid):
        s_i = np.zeros_like(self.meta_strategies[aid], np.float32)
        s_i[sid] = 1
        u_sigma = compute_utility(payoff, self.meta_strategies[:aid], True)
        u_sigma = compute_utility(u_sigma, self.meta_strategies[aid+1:])
        assert u_sigma.shape == s_i.shape, u_sigma
        u_pi = s_i @ u_sigma
        u_sigma = self.meta_strategies[aid] @ u_sigma
        self.regrets[aid][sid] = self.regrets[aid][sid] + u_pi - u_sigma

    def update_meta_strategy(self, aid):
        r_plus = np.maximum(0, self.regrets[aid])
        r_plus_total = sum(r_plus)
        self.n_meta_strategy_updates[aid] += 1
        self.meta_strategies[aid] = r_plus / r_plus_total \
            if r_plus_total > 0 else \
                np.ones(self.meta_strategies[aid].size, np.float32) / self.meta_strategies[aid].size
        self.average_meta_strategies[aid] += 1/self.n_meta_strategy_updates[aid] \
            * (self.meta_strategies[aid] - self.average_meta_strategies[aid])

    def compute(self, payoffs, n_agents, n_strategies, n_iterations):
        for payoff in payoffs:
            assert payoff.shape == tuple([n_strategies for _ in range(n_agents)]), (payoff.shape, n_agents)
        for t in range(n_iterations):
            for aid in range(n_agents):
                for sid in range(n_strategies):
                    self.update_regrets(payoffs[aid], aid, sid)
            for aid in range(n_agents):
                self.update_meta_strategy(aid)

        return self.average_meta_strategies


# if __name__ == '__main__':
#     import nashpy as nash

#     m = 5
#     n = 2
#     A = np.random.randint(0, 10, size=(5, 5))
#     payoffs = [A, -A]
#     game = nash.Game(A)
#     print(game.lemke_howson(initial_dropped_label=0))
#     rm = RegretMatching(n, m)
#     ans = rm.compute(payoffs, n, m, 10000)
#     print(ans)
