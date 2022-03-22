import os

from utility.utils import config_attr
from utility import yaml_op


class CheckpointBase:
    def save(self):
        raise NotImplementedError

    def restore(self):
        if os.path.exists(self._path):
            config = yaml_op.load(self._path)
            if config is not None:
                config_attr(self, config, config_as_attr=False, private_attr=True)
