import time
import datetime
import collections
import functools

from tools.log import do_logging
from core.typing import AttrDict
from tools.aggregator import Aggregator


def time2str(duration):
  if duration < 1e-1:
    return f'{duration*1000:.3g}ms'
  elif duration < 60:
    return f'{duration:.3g}s'
  elif duration < 3600:
    return f'{duration/60:.3g}m'
  elif duration < 86400:
    return f'{duration/3600:.3g}h'
  else:
    return f'{duration/86400:.3g}d'


def get_current_datetime():
  t = int(time.time())
  t = datetime.datetime.fromtimestamp(t)
  return t


def compute_time_left(elapsed_time, curr_step, remaining_steps):
  if remaining_steps <= 0:
    return datetime.timedelta(0)
  time_left = remaining_steps / curr_step * elapsed_time
  return time_left


def timeit_now(func, *args, name=None, to_print=True, 
    return_duration=False, **kwargs):
  start_time = time.gmtime()
  start = time.time()
  result = func(*args, **kwargs)
  end = time.time()
  end_time = time.gmtime()

  if to_print:
    do_logging(f'{name if name else func.__name__}: '
      f'Start "{time.strftime("%d %b %H:%M:%S", start_time)}"' 
      f'End "{time.strftime("%d %b %H:%M:%S", end_time)}" ' 
      f'Duration "{time2str(end - start)}s"')

  return end - start, result if return_duration else result


def timeit(func, **timer_kwargs):
  Timer.aggregators[func.__name__]
  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    with Timer(func.__name__, **timer_kwargs):
      return func(*args, **kwargs)
  return wrapper


def _get_time_prefix(prefix=None):
  if prefix:
    prefix = f'time/{prefix}'
  else:
    prefix = f'time'
  return prefix


class Timer:
  aggregators = collections.defaultdict(Aggregator)

  def __init__(self, summary_name, period=None, mode='average', to_record=True, min_duration=1e-4):
    self._to_record = to_record
    if self._to_record:
      self._min_duration = min_duration
      self.aggregators[summary_name]
      self._summary_name = summary_name
      self._period = period
      assert mode in ['average', 'sum']
      self._mode = mode

  @property
  def name(self):
    return self._summary_name

  def __enter__(self):
    if self._to_record:
      self._start = time.time()
    return self
  
  def __exit__(self, exc_type, exc_value, traceback):
    if self._to_record:
      duration = time.time() - self._start
      aggregator = self.aggregators[self._summary_name]
      if duration > self._min_duration:
        aggregator.add(duration)
      if self._period is not None:
        name = self._summary_name
        if self._period == 1:
          do_logging(f'{name} duration: "{time2str(duration)}"',backtrack=3)
        elif self._mode == 'average':
          duration = aggregator.average()
          do_logging(
            f'{name} duration: "{time2str(duration)}" averaged over {self._period} times',
            backtrack=3
          )
          # aggregator.reset()
        else:
          duration = aggregator.sum
          do_logging(
            f'{name} duration: "{time2str(duration)}" for {aggregator.count} times', 
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
  
  def count(self):
    return self.aggregators[self._summary_name].count
  
  def to_stats(self, prefix=None):
    prefix = _get_time_prefix(prefix)
    return {
      f'{prefix}/{self.name}_total': self.total(), 
      f'{prefix}/{self.name}': self.average(), 
      f'{prefix}/{self.name}_count': self.count()
    }

  @staticmethod
  def all_stats(prefix=None):
    prefix = _get_time_prefix(prefix)
    stats = AttrDict()
    for k, v in Timer.aggregators.items():
      stats[f'{prefix}/{k}_total'] = v.total
      stats[f'{prefix}/{k}'] = v.average()
      stats[f'{prefix}/{k}_count'] = v.count
    return stats

  @staticmethod
  def sorted_stats(prefix=None):
    prefix = _get_time_prefix(prefix)
    stats_list = sorted([(k, v) for k, v in Timer.aggregators.items()], 
      key=lambda x: x[1].total, reverse=True)
    stats = AttrDict()
    for i, (k, v) in enumerate(stats_list):
      stats[f'{prefix}/{i:02d}/{k}_total'] = v.total
      stats[f'{prefix}/{i:02d}/{k}'] = v.average()
      stats[f'{prefix}/{i:02d}/{k}_count'] = v.count
    return stats

  @staticmethod
  def top_stats(prefix=None, n=15):
    prefix = _get_time_prefix(prefix)
    stats_list = sorted([(k, v) for k, v in Timer.aggregators.items()], 
      key=lambda x: x[1].total, reverse=True)
    if n is not None:
      stats_list = stats_list[:n]
    stats = AttrDict()
    for k, v in stats_list:
      stats[f'{prefix}/{k}_total'] = v.total
      stats[f'{prefix}/{k}'] = v.average()
      stats[f'{prefix}/{k}_count'] = v.count
    return stats


class TBTimer:
  aggregators = collections.defaultdict(Aggregator)

  def __init__(self, summary_name, period=1, to_record=True, print_terminal_info=False):
    self._to_record = to_record
    if self._to_record:
      self._summary_name = summary_name
      self._period = period
      self._print_terminal_info = print_terminal_info

  def __enter__(self):
    if self._to_record:
      self._start = time.time()
    return self
  
  def __exit__(self, exc_type, exc_value, traceback):
    import tensorflow as tf
    if self._to_record:
      duration = time.time() - self._start
      aggregator = self.aggregators[self._summary_name]
      aggregator.add(duration)
      if aggregator.count >= self._period:
        duration = aggregator.average()
        step = tf.summary.experimental.get_step()
        tf.summary.scalar(f'time/{self._summary_name}', duration, step=step)
        aggregator.reset()
        if self._print_terminal_info:
          do_logging(f'{self._summary_name} duration: "{time2str(duration)}" averaged over {self._period} times')


class LoggerTimer:
  def __init__(self, logger, summary_name, to_record=True):
    self._to_record = to_record
    if self._to_record:
      self._logger = logger
      self._summary_name = summary_name

  def __enter__(self):
    if self._to_record:
      self._start = time.time()
    return self
  
  def __exit__(self, exc_type, exc_value, traceback):
    if self._to_record:
      duration = time.time() - self._start
      self._logger.store(**{self._summary_name: duration})


class Every:
  def __init__(self, period, start=0, init_next=False, final=None):
    self._period = period
    self._curr = start
    self._next = start + period if init_next and period is not None else start
    self._final = final
    self._diff = 0
  
  def __call__(self, step):
    self._diff = step - self._curr
    if self._period is None:
      return False
    if step >= self._next or (self._final is not None and step >= self._final):
      self._curr = self._next
      self._next = self._curr + self._period
      return True
    return False

  def difference(self):
    """ Compute the most recent update difference """
    return self._diff

class NamedEvery:
  named_everys = {}

  def __init__(self, name, period, start=0, init_next=False, final=None):
    if name not in self.named_everys:
      self.named_everys[name] = Every(
        period, start, init_next=init_next, final=final)
    self._every = self.named_everys[name]
  
  def __call__(self, step):
    return self._every(step)
  
  def difference(self):
    return self._every.difference()
