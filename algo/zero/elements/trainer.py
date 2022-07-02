import functools
import tensorflow as tf

from core.elements.trainer import Trainer as TrainerBase, create_trainer
from core.decorator import override
from core.optimizer import create_optimizer
from core.tf_config import build
from utility.display import print_dict
from utility import pkg
from optimizers.adam import Adam
from optimizers.rmsprop import RMSprop
from utility.utils import dict2AttrDict


class Trainer(TrainerBase):
    def construct_optimizers(self):
        keys = sorted([
            k for k in self.model.keys() if not k.startswith('target')
        ])
        modules = tuple([
            self.model[k] for k in keys if not k.startswith('meta')
        ])
        config = dict2AttrDict(self.config, to_copy=True)
        opts = {
            'adam': Adam, 
            'rmsprop': RMSprop
        }
        config.optimizer.opt_name = opts[config.optimizer.opt_name]
        self.optimizer = create_optimizer(
            modules, config.optimizer
        )
        
        if self.config.get('meta_opt'):
            meta_modules = tuple([
                self.model[k] for k in keys if k.startswith('meta')
            ])
            self.meta_opt = create_optimizer(
                meta_modules, self.config.meta_opt
            )

    @override(TrainerBase)
    def _build_train(self, env_stats):
        algo = self.config.algorithm.split('-')[-1]
        get_data_format = pkg.import_module(
            'elements.utils', algo=algo).get_data_format
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = get_data_format(
            self.config, env_stats, self.loss.model)
        print_dict(TensorSpecs, prefix='Tensor Specifications')
        self.train = build(self.train, TensorSpecs)
        return True

    def raw_train(
        self, 
        obs, 
        action, 
        reward, 
        discount, 
        reset, 
        mu_prob, 
        mu=None, 
        mu_mean=None, 
        mu_std=None, 
        state=None, 
        mask=None
    ):
        tape, loss, terms = self.loss.loss(
            obs=obs, 
            action=action, 
            reward=reward, 
            discount=discount, 
            reset=reset, 
            mu_prob=mu_prob, 
            mu=mu, 
            mu_mean=mu_mean, 
            mu_std=mu_std, 
            state=state, 
            mask=mask
        )

        terms['norm'], terms['var_norm'], grads = \
            self.optimizer(tape, loss, return_var_norms=True, return_grads=True)
        terms['grads_norm'] = tf.linalg.global_norm(list(grads.values()))
        
        return terms

    def raw_meta_train(
        self, 
        obs, 
        action, 
        reward, 
        discount, 
        reset, 
        mu_prob, 
        mu=None, 
        mu_mean=None, 
        mu_std=None, 
        state=None, 
        mask=None
    ):
        with tf.GradientTape(persistent=True):
            for _ in range(self.config.inner_steps):
                tape, loss, terms = self.loss.loss(
                    obs=obs, 
                    action=action, 
                    reward=reward, 
                    discount=discount, 
                    reset=reset, 
                    mu_prob=mu_prob, 
                    mu=mu, 
                    mu_mean=mu_mean, 
                    mu_std=mu_std, 
                    state=state, 
                    mask=mask, 
                )

                terms['norm'], terms['var_norm'], grads = \
                    self.optimizer(tape, loss, return_var_norms=True, return_grads=True)
                terms['grads_norm'] = tf.linalg.global_norm(list(grads.values()))
            
        
        return terms

create_trainer = functools.partial(create_trainer,
    name='zero', trainer_cls=Trainer
)
