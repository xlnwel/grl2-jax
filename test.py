import logging
from core.log import do_logging, get_sys_logger
from utility.timer import timeit

logging.basicConfig(level=logging.INFO, 
    format=f'%(asctime)s: %(levelname)s: %(name)s: %(message)s',
)

d = {
    'a': {
        1: 2,
        2: 3,
    },
    'b': 'abc'
}

timeit(do_logging, d)
do_logging(d)