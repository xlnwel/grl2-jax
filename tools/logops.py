import os
from enum import Enum
from datetime import datetime, timedelta


class DirLevel(Enum):
    ROOT = 0
    LOGS = 1
    ENV = 2
    ALGO = 3
    DATE = 4
    MODEL = 5
    SEED = 6
    FINAL = 7
    UNKNOWN = 8
    
    def next(self):
        v = self.value + 1
        if self.greater(DirLevel.FINAL):
            raise ValueError(f'Enumeration ended')
        return DirLevel(v)

    def greater(self, other):
        return self.value > other.value
    
    def __eq__(self, other: object) -> bool:
        return self.value == other.value


def get_level(search_dir, last_prefix=None):
    for d in os.listdir(search_dir):
        if d.endswith('logs'):
            return DirLevel.ROOT
    all_names = search_dir.split('/')
    last_name = all_names[-1]
    if last_prefix is not None:
        if not isinstance(last_prefix, (list, tuple)):
            last_prefix = [last_prefix]
        if any([last_name.startswith(p) for p in last_prefix]):
            return DirLevel.FINAL
    if last_name.endswith('logs'):
        return DirLevel.LOGS
    if last_name.startswith('seed'):
        return DirLevel.SEED
    suite = None
    for name in search_dir.split('/'):
        if name.endswith('-logs'):
            suite = name.split('-')[0]
    if last_name.startswith(f'{suite}'):
        return DirLevel.ENV
    if last_name.isdigit():
        return DirLevel.DATE
    # find algorithm name
    algo = None
    for i, name in enumerate(all_names):
        if name.isdigit():
            algo = all_names[i-1]
            if len(all_names) == i+2:
                return DirLevel.MODEL
    if algo is None:
        return DirLevel.ALGO
    
    return DirLevel.FINAL


def get_date(args_date):
    date = set()
    for d in args_date:
        if d.isdigit():
            date.add(str(d))
        else:
            dt = datetime.now()
            if d == 'today':
                dt = dt
            elif d == 'tomorrow':
                # yesterday
                dt = dt + timedelta(days=1)
            elif d == 'yd':
                # tomorrow
                dt = dt - timedelta(days=1)
            elif d == 'dby':
                # the day before yesterday
                dt = dt - timedelta(days=2)
            elif d == 'tda':
                # three days ago
                dt = dt - timedelta(days=3)
            elif d == 'fda':
                # four days ago
                dt = dt - timedelta(days=4)
            date.add(dt.strftime('%m%d'))
    return date


def fixed_pattern_search(
    search_dir, 
    level=DirLevel.LOGS, 
    env=None, 
    algo=None, 
    date=None, 
    model=None, 
    final_level=DirLevel.FINAL, 
    final_name=None
):
    if level == final_level:
        yield search_dir
    elif not os.path.isdir(search_dir) or level.greater(final_level):
        return []
    elif level == final_level:
        last_name = search_dir.split('/')[-1]
        if not isinstance(final_name, (list, tuple)):
            final_name = [final_name]
        if final_name is None or any([last_name.startswith(p) for p in final_name]):
            yield search_dir
        else:
            return []
    else:
        if level == DirLevel.MODEL and model:
            if all([m not in search_dir for m in model]):
                return []
        elif level == DirLevel.DATE and date:
            if all([d not in search_dir for d in date]):
                return []
        elif level == DirLevel.ALGO and algo:
            if all([not search_dir.endswith(a) for a in algo]):
                return []
        elif level == DirLevel.ENV and env:
            if all([e not in search_dir for e in env]):
                return []
        for d in os.listdir(search_dir):
            for f in fixed_pattern_search(
                os.path.join(search_dir, d), 
                level=level.next(), 
                env=env, 
                algo=algo, 
                date=date, 
                model=model, 
                final_level=final_level, 
                final_name=final_name
            ):
                yield f

