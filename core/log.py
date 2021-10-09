import os
import logging

from utility.utils import get_frame


""" Logging operations """
def get_sys_logger(backtrack=1):
    frame = get_frame(backtrack)
    filename = frame.f_code.co_filename
    filename = filename.replace(f'{os.getcwd()}/', '')
    filename = filename.replace('.py', '')
    filename = filename.replace('/', '.')
    logger = logging.getLogger(filename)
    return logger

def do_logging(x, prefix='', logger=None, level='INFO', func_lineno=None, backtrack=2):
    if logger is None:
        logger = get_sys_logger(backtrack)
    frame = get_frame(backtrack)
    if func_lineno is None:
        funcname = frame.f_code.co_name
        lineno = frame.f_lineno
        func_lineno = f'{funcname}: {lineno}: '
    new_prefix = func_lineno + prefix
    log_func = {
        'critical': logger.critical,
        'error': logger.error,
        'warning': logger.warning,
        'info': logger.info,
        'debug': logger.debug,
        'print': print,
    }[level.lower()]

    if isinstance(x, str):
        log_func(f'{new_prefix}{x}')
    elif isinstance(x, (list, tuple)):
        for v in x:
            if isinstance(v, dict):
                do_logging(v, logger=logger, prefix=prefix+'\t', func_lineno=func_lineno)
            else:
                log_func(f'{new_prefix}{v}')
    elif isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, dict):
                log_func(f'{new_prefix}{k}')
                do_logging(v, logger=logger, prefix=prefix+'\t', func_lineno=func_lineno)
            else:
                log_func(f'{new_prefix}{k}: {v}')
    else:
        raise ValueError(f'{x} is of unknown type.')
