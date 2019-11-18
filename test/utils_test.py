import numpy as np

from utility.utils import standardize


class TestClass:
    def test_standardize(self):
        epsilon = 1e-5
        x = np.arange(1000).reshape((2, 50, 2, 5))
        mask = np.ones(100).reshape((2, 50, 1))
        mask[1, 14:] = 0
        result = standardize(x, epsilon=epsilon, mask=mask).flatten()
        idx = sum(result != 0)
        result = result[:idx]
        x_masked = x.flatten()[:idx]
        base = standardize(x_masked, epsilon=epsilon)

        assert np.all(result-epsilon < base)
        assert np.all(base < result+epsilon)
