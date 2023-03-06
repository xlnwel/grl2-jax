import os
import logging

from tools.file import source_file
from algo.dynamics.elements.model import Model as ModelBase, \
    setup_config_from_envstats, construct_fake_data
from .utils import *

# register ppo-related networks 
source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))
logger = logging.getLogger(__name__)


class Model(ModelBase):
    def build_nets(self):
        aid = self.config.get('aid', 0)
        self.is_action_discrete = self.env_stats.is_action_discrete[aid]
        data = construct_fake_data(self.env_stats, aid)

        self.params.model, self.modules.model = self.build_net(
            data.obs[:, 0, :], data.action[:, 0], name='model')
        self.params.emodels, self.modules.emodels = self.build_net(
            data.obs, data.action, True, name='emodels')
        self.params.reward, self.modules.reward = self.build_net(
            data.obs, data.action, name='reward')
        self.params.discount, self.modules.discount = self.build_net(
            data.obs, name='discount')


def create_model(
    config, 
    env_stats, 
    name='model', 
    **kwargs
): 
    config = setup_config_from_envstats(config, env_stats)

    return Model(
        config=config, 
        env_stats=env_stats, 
        name=name,
        **kwargs
    )
