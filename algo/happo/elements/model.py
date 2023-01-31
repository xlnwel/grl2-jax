import os
import math
import logging
import jax
from jax import lax, nn, random
import jax.numpy as jnp
import haiku as hk

from core.log import do_logging
from core.typing import AttrDict, dict2AttrDict
from tools.file import source_file
from jax_tools import jax_dist, jax_utils
from tools.display import print_dict_info
from algo.zero.elements.model import Model as ModelBase, setup_config_from_envstats
from algo.zero.elements.utils import get_initial_state

# register ppo-related networks 
source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))
logger = logging.getLogger(__name__)


def construct_fake_data(env_stats, aid, batch_size=1):
    basic_shape = (batch_size, 1, len(env_stats.aid2uids[aid]))
    shapes = env_stats.obs_shape[aid]
    dtypes = env_stats.obs_dtype[aid]
    action_dim = env_stats.action_dim[aid]
    data = {k: jnp.zeros((*basic_shape, *v), dtypes[k]) 
        for k, v in shapes.items()}
    data = dict2AttrDict(data)
    data.setdefault('global_state', data.obs)
    data.setdefault('hidden_state', data.obs)
    data.action = jnp.zeros((*basic_shape, action_dim), jnp.float32)
    data.state_reset = jnp.zeros(basic_shape, jnp.float32)

    # print_dict_info(data)

    return data

class Model(ModelBase):
    def compile_model(self):
        super().compile_model()
        self.jit_action_logprob = jax.jit(self.action_logprob, static_argnames=('evaluation'))

    def action_logprob(
        self,
        params,
        rng,
        data,
    ):
        rngs = random.split(rng, 3)
        state_reset, _ = jax_utils.split_data(
            data.state_reset, axis=1)
        policy_state = None if data.state is None else \
            get_initial_state(data.state.policy, 0)
        act_out, policy_state = self.modules.policy(
            params.policy, 
            rngs[0], 
            data.obs, 
            state_reset, 
            policy_state, 
            action_mask=data.action_mask, 
        )
        act_dist = self.policy_dist(act_out, False)
        logprob = act_dist.log_prob(data.action)

        return logprob

    def compute_value(self, data):
        @jax.jit
        def comp_value(params, rng, global_state, state_reset=None, state=None):
            v, _ = self.modules.value(
                params.value, rng, 
                global_state, state_reset, state
            )
            return v
        self.act_rng, rng = random.split(self.act_rng)
        value = comp_value(self.params, rng, **data)
        return value


def create_model(
    config, 
    env_stats, 
    name='happo', 
    **kwargs
): 
    config = setup_config_from_envstats(config, env_stats)

    return Model(
        config=config, 
        env_stats=env_stats, 
        name=name,
        **kwargs
    )


if __name__ == '__main__':
    from tools.yaml_op import load_config
    from env.func import create_env
    from tools.display import pwc
    config = load_config('algo/zero_mr/configs/magw_a2c')
    
    env = create_env(config.env)
    model = create_model(config.model, env.stats())
    data = construct_fake_data(env.stats(), 0)
    print(model.action(model.params, data))
    pwc(hk.experimental.tabulate(model.raw_action)(model.params, data), color='yellow')
