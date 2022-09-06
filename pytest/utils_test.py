import numpy as np

from tools.utils import *


class TestClass:
    def test_moments(self):
        # moments along all dimensions
        for x in [np.random.randn(10), np.random.randn(10, 3)]:
            mask = np.random.randint(2, size=x.shape[:1])
            mean, std = moments(x, mask=mask)
            x2 = []
            for i, m in enumerate(mask):
                if m == 1:
                    x2.append(x[i])
            x2 = np.array(x2)
            print(np.sum(x2, axis=0), np.sum(mask))
            np.testing.assert_allclose(mean, np.mean(x2))
            np.testing.assert_allclose(std, np.std(x2))

        for x in [np.random.randn(10), np.random.randn(10, 3)]:
            mask = np.random.randint(2, size=x.shape[:1])
            mean, std = moments(x, axis=0, mask=mask)
            x2 = []
            for i, m in enumerate(mask):
                if m == 1:
                    x2.append(x[i])
            x2 = np.array(x2)
            print(np.sum(x2, axis=0), np.sum(mask))
            np.testing.assert_allclose(mean, np.mean(x2, axis=0))
            np.testing.assert_allclose(std, np.std(x2, axis=0))

    def test_standardize(self):
        epsilon = 1e-5
        x = np.arange(1000).reshape((2, 50, 2, 5))
        mask = np.ones(100).reshape((2, 50, 1))
        mask[1, 14:] = 0
        result = standardize(x, mask=mask, epsilon=epsilon).flatten()
        idx = sum(result != 0)
        result = result[:idx]
        x_masked = x.flatten()[:idx]
        base = standardize(x_masked, epsilon=epsilon)

        np.testing.assert_allclose(result, base)
