import os
import logging
from datetime import datetime

from core.log import colorize, pwt, pwc, pwtc
from tools.utils import get_frame


""" Logging operations """
def setup_logging(verbose):
    verbose = getattr(logging, verbose.upper())
    logging.basicConfig(
        level=verbose, 
        format=f'%(asctime)s: %(levelname)s: %(name)s: %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )

def get_sys_logger(backtrack=1):
    frame = get_frame(backtrack)
    filename = frame.f_code.co_filename
    filename = filename.replace(f'{os.getcwd()}/', '')
    filename = filename.replace('.py', '')
    filename = filename.replace('/', '.')
    logger = logging.getLogger(filename)
    return logger

def do_logging(
    x, 
    prefix='', 
    logger=None, 
    level='INFO', 
    func_lineno=None, 
    backtrack=2, 
    time=False, 
    color=None, 
    bold=False, 
    highlight=False, 
    **log_kwargs, 
):
    if logger is None:
        logger = get_sys_logger(backtrack)
    frame = get_frame(backtrack)
    if func_lineno is None:
        filename = frame.f_code.co_filename
        filename = filename.rsplit('/', 1)[-1]
        funcname = frame.f_code.co_name
        lineno = frame.f_lineno
        func_lineno = f'{filename} {funcname}: line {lineno}: '
    if prefix:
        prefix = f'{prefix}: '
    new_prefix = func_lineno + prefix
    if time:
        new_prefix = f'{datetime.now()}: {new_prefix}'
    log_func = {
        'critical': logger.critical,
        'error': logger.error,
        'warning': logger.warning,
        'info': logger.info,
        'debug': logger.debug,
        'print': print,
        'pwt': pwt, 
        'pwc': pwc, 
        'pwtc': pwtc
    }[level.lower()]

    def decorate_content(x):
        content = f'{new_prefix}: {x}'
        if color or bold or highlight:
            content = colorize(content, color=color, bold=bold, highlight=highlight)
        return content

    if isinstance(x, str):
        log_func(decorate_content(x))
    elif isinstance(x, (list, tuple)):
        for v in x:
            if isinstance(v, dict):
                do_logging(
                    v, 
                    logger=logger, 
                    prefix=prefix, 
                    func_lineno=func_lineno, 
                    time=time, 
                    color=color, 
                    bold=bold, 
                    highlight=highlight
                )
            else:
                log_func(decorate_content(v))
    elif isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, dict):
                log_func(decorate_content(k))
                do_logging(
                    v, 
                    logger=logger, 
                    prefix=prefix, 
                    func_lineno=func_lineno, 
                    time=time, 
                    color=color, 
                    bold=bold, 
                    highlight=highlight
                )
            else:
                log_func(decorate_content(f'{k}: {v}'))
    else:
        raise ValueError(f'{x} is of unknown type.')
