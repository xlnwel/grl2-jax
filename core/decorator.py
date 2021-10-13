import functools
import logging

from utility.utils import config_attr


logger = logging.getLogger(__name__)


def config(init_fn):
    """ Adds config to attr """
    @functools.wraps(init_fn)
    def wrapper(self, config, *args, name=None, **kwargs):
        config_attr(self, config)
        if name is not None:
            self.name = name

        if 'model_name' in config:
            self._model_name = config['model_name'] or 'baseline'

        init_fn(self, *args, **kwargs)

    return wrapper

def override(cls):
    @functools.wraps(cls)
    def check_override(method):
        if method.__name__ not in dir(cls):
            raise NameError("{} does not override any method of {}".format(
                method, cls))
        return method

    return check_override
