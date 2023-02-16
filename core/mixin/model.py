import logging
import jax

from core.log import *


logger = logging.getLogger(__name__)


def update_params(source, target, polyak):
    return jax.tree_util.tree_map(
        lambda x, y: polyak * x + (1.-polyak) * y, target, source)
