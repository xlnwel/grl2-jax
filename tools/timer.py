from time import strftime, gmtime, time
from collections import defaultdict

from core.log import do_logging
from core.typing import AttrDict
from tools.aggregator import Aggregator


def timeit(func, *args, name=None, to_print=True, 
        return_duration=False, **kwargs):
    start_time = gmtime()
    start = time()
    result = func(*args, **kwargs)
    end = time()
    end_time = gmtime()

    if to_print:
        do_logging(f'{name if name else func.__name__}: '
            f'Start "{strftime("%d %b %H:%M:%S", start_time)}"' 
            f'End "{strftime("%d %b %H:%M:%S", end_time)}" ' 
            f'Duration "{end - start:.3g}s"')

    return end - start, result if return_duration else result


class Timer:
    aggregators = defaultdict(Aggregator)

    def __init__(self, summary_name, period=None, mode='average', to_record=True):
        self._to_log = to_record
        if self._to_log:
            self._summary_name = summary_name
            self._period = period
            assert mode in ['average', 'sum']
            self._mode = mode

    @property
    def name(self):
        return self._summary_name

    def __enter__(self):
        if self._to_log:
            self._start = time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self._to_log:
            duration = time() - self._start
            aggregator = self.aggregators[self._summary_name]
            aggregator.add(duration)
            if self._period is not None and aggregator.count >= self._period:
                if self._mode == 'average':
                    duration = aggregator.average()
                    duration = (f'{duration*1000:.3g}ms' if duration < 1e-1 
                                else f'{duration:.3g}s')
                    do_logging(
                        f'{self._summary_name} duration: "{duration}" averaged over {self._period} times',
                        backtrack=3
                    )
                    aggregator.reset()
                else:
                    duration = aggregator.sum
                    do_logging(
                        f'{self._summary_name} duration: "{duration}" for {aggregator.count} times', 
                            backtrack=3
                    )

    def reset(self):
        aggregator = self.aggregators[self._summary_name]
        aggregator.reset()
    
    def average(self):
        return self.aggregators[self._summary_name].average()
        
    def last(self):
        return self.aggregators[self._summary_name].last
    
    def total(self):
        return self.aggregators[self._summary_name].total
    
    def to_stats(self, prefix=None):
        if prefix:
            prefix = f'timer/{prefix}/'
        else:
            prefix = f'timer/'
        return {
            f'{prefix}/{self.name}_total': self.total(),
            f'{prefix}/{self.name}': self.average()
        }

    @staticmethod
    def all_stats(prefix=None):
        if prefix:
            prefix = f'time/{prefix}'
        else:
            prefix = f'time'
        stats = AttrDict()
        for k, v in Timer.aggregators.items():
            stats[f'{prefix}/{k}_total'] = v.total
            stats[f'{prefix}/{k}'] = v.average()
        return stats


class TBTimer:
    aggregators = defaultdict(Aggregator)

    def __init__(self, summary_name, period=1, to_record=True, print_terminal_info=False):
        self._to_log = to_record
        if self._to_log:
            self._summary_name = summary_name
            self._period = period
            self._print_terminal_info = print_terminal_info

    def __enter__(self):
        if self._to_log:
            self._start = time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        import tensorflow as tf
        if self._to_log:
            duration = time() - self._start
            aggregator = self.aggregators[self._summary_name]
            aggregator.add(duration)
            if aggregator.count >= self._period:
                duration = aggregator.average()
                step = tf.summary.experimental.get_step()
                tf.summary.scalar(f'time/{self._summary_name}', duration, step=step)
                aggregator.reset()
                if self._print_terminal_info:
                    do_logging(f'{self._summary_name} duration: "{duration}" averaged over {self._period} times')


class LoggerTimer:
    def __init__(self, logger, summary_name, to_record=True):
        self._to_log = to_record
        if self._to_log:
            self._logger = logger
            self._summary_name = summary_name

    def __enter__(self):
        if self._to_log:
            self._start = time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self._to_log:
            duration = time() - self._start
            self._logger.store(**{self._summary_name: duration})


class Every:
    def __init__(self, period, start=0, final=None):
        self._period = period
        self._curr = start
        self._next = start
        self._final = final
        self._diff = 0
    
    def __call__(self, step):
        self._diff = step - self._curr
        if self._period is None:
            return False
        if step >= self._next or (self._final is not None and step >= self._final):
            self._curr = step
            self._next = step + self._period
            return True
        return False

    def difference(self):
        """ Compute the most recent update difference """
        return self._diff
