import collections
import numpy as np

from utility.utils import expand_dims_match, moments


Stats = collections.namedtuple('RMS', 'mean var count')


def merge_rms(rms1, rms2):
    mean1, var1, count1 = rms1
    mean2, var2, count2 = rms2
    delta = mean2 - mean1
    total_count = count1 + count2

    new_mean = mean1 + delta * count2 / total_count
    # no minus one here to be consistent with np.std
    m_a = var1 * count1
    m_b = var2 * count2
    M2 = m_a + m_b + delta**2 * count1 * count2 / total_count
    assert np.all(np.isfinite(M2)), f'M2: {M2}'
    new_var = M2 / total_count

    return new_mean, new_var, total_count


class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, axis, epsilon=1e-8, clip=None, name=None, ndim=None):
        """ Computes running mean and std from data
        A reimplementation of RunningMeanStd from OpenAI's baselines

        Args:
            axis: axis along which we compute mean and std from incoming data. 
                If it's None, we only receive at a time a sample without batch dimension
            ndim: expected number of dimensions for the stats, useful for debugging
        """
        self.name = name

        if isinstance(axis, int):
            axis = (axis, )
        elif isinstance(axis, (tuple, list)):
            axis = tuple(axis)
        elif axis is None:
            pass
        else:
            raise ValueError(f'Invalid axis({axis}) of type({type(axis)})')

        if isinstance(axis, tuple):
            assert axis == tuple(range(len(axis))), \
                f'Axis should only specifies leading axes so that '\
                f'mean and var can be broadcasted automatically when normalizing. '\
                f'But receving axis = {axis}'
        self._axis = axis
        if self._axis is not None:
            self._shape_slice = np.s_[: max(self._axis)+1]
        self._mean = 0
        self._var = 0
        self._epsilon = epsilon
        self._count = epsilon
        self._clip = clip
        self._ndim = ndim # expected number of dimensions o

    @property
    def axis(self):
        return self._axis

    def set_rms_stats(self, mean, var, count):
        self._mean = mean
        self._var = var
        self._std = np.sqrt(self._var)
        self._count = count

    def get_rms_stats(self):
        return Stats(self._mean, self._var, self._count)

    def update(self, x, mask=None):
        x = x.astype(np.float64)
        if self._axis is None:
            assert mask is None, mask
            batch_mean, batch_var, batch_count = x, np.zeros_like(x), 1
        else:
            batch_mean, batch_var = moments(x, self._axis, mask)
            batch_count = np.prod(x.shape[self._shape_slice]) \
                if mask is None else np.sum(mask)
        if batch_count > 0:
            if self._ndim is not None:
                assert batch_mean.ndim == self._ndim, (batch_mean.shape, self._ndim)
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        if self._count == self._epsilon:
            self._mean = np.zeros_like(batch_mean, 'float64')
            self._var = np.ones_like(batch_var, 'float64')
        assert self._mean.shape == batch_mean.shape
        assert self._var.shape == batch_var.shape

        new_mean, new_var, total_count = merge_rms(
            (self._mean, self._var, self._count), 
            (batch_mean, batch_var, batch_count))
        self._mean = new_mean
        self._var = new_var
        self._std = np.sqrt(self._var)
        self._count = total_count
        assert np.all(self._var > 0), self._var[self._var <= 0]

    def normalize(self, x, zero_center=True, mask=None):
        assert not np.isinf(np.std(x)), f'{np.min(x)}\t{np.max(x)}'
        assert self._var is not None, (self._mean, self._var, self._count)
        assert x.ndim == self._var.ndim + (0 if self._axis is None else len(self._axis)), \
            (x.shape, self._var.shape, self._axis)
        if mask is not None:
            assert mask.ndim == len(self._axis), (mask.shape, self._axis)
            old = x.copy()
        if zero_center:
            x -= self._mean
        x /= self._std
        if self._clip:
            x = np.clip(x, -self._clip, self._clip)
        if mask is not None:
            mask = expand_dims_match(mask, x)
            x = np.where(mask, x, old)
        x = x.astype(np.float32)
        return x
