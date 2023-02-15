import logging
import jax

from core.log import *


logger = logging.getLogger(__name__)


def sync_params(source, target, filters=[]):
    for k, v in source.items():
        if k not in filters:
            target[k] = v


def update_params(source, target, polyak):
    return jax.tree_util.tree_map(
        lambda x, y: polyak * x + (1.-polyak) * y, target, source)
