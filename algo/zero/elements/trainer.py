import functools
import tensorflow as tf

from core.elements.trainer import Trainer as TrainerBase, create_trainer
from core.decorator import override
from core.log import do_logging
from core.optimizer import create_optimizer
from core.tf_config import build
from utility import pkg
from optimizers.adam import Adam
from optimizers.rmsprop import RMSprop
from utility.meta import compute_meta_gradients
from utility.utils import dict2AttrDict


def _add_norm(terms, d, tag, suffix, norm_name):
    terms.update({
        f'{tag}/{k}_{suffix}': v for k, v in d.items()
    })
    if norm_name is not None:
        terms[f'{tag}/{norm_name}'] = tf.linalg.global_norm(list(d.values()))
    return terms


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
        config.meta_opt.opt_name = opts[config.meta_opt.opt_name]
        self.optimizer = create_optimizer(
            modules, config.optimizer
        )
        
        if self.config.inner_steps is not None:
            self.meta_modules = tuple([
                self.model[k] for k in keys if k.startswith('meta')
            ])
            self.meta_opt = create_optimizer(
                self.meta_modules, config.meta_opt
            )

    def ckpt_model(self):
        if self.config.inner_steps is not None:
            models = {
                f'{self._raw_name}_opt': self.optimizer, 
                f'{self._raw_name}_meta_opt': self.meta_opt, 
            }
        else:
            models = super().ckpt_model()
        return models

    @override(TrainerBase)
    def _build_train(self, env_stats):
        algo = self.config.algorithm.split('-')[-1]
        get_data_format = pkg.import_module(
            'elements.utils', algo=algo).get_data_format
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = get_data_format(
            self.config, env_stats, self.loss.model, meta=False)
        do_logging(TensorSpecs, prefix='Tensor Specifications', level='print')
        self.train = build(self.train, TensorSpecs)
        if self.config.inner_steps is not None:
            meta_train = tf.function(self.raw_meta_train)
            TensorSpecs = get_data_format(
                self.config, env_stats, self.loss.model, meta=True)
            do_logging(TensorSpecs, prefix='Meta Tensor Specifications', level='print')
            self.meta_train = build(meta_train, TensorSpecs)
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
        with tf.GradientTape() as tape:
            loss, terms = self.loss.loss(
                tape, 
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
                debug=self.config.inner_steps is None
            )

        if self.config.inner_steps is None:
            terms['grads_norm'], var_norms, grads = self.optimizer(
                tape, 
                loss, 
                return_var_norms=True, 
                return_grads=True
            )
            terms['var_norm'] = list(var_norms.values())
            terms['clipped_grads_norm'] = tf.linalg.global_norm(list(grads.values()))
        else:
            self.optimizer(tape, loss)
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
        with tf.GradientTape(persistent=True) as meta_tape:
            grads_list = []
            for i in range(self.config.inner_steps):
                with tf.GradientTape() as tape:
                    loss, terms = self.loss.loss(
                        tape=tape, 
                        obs=obs[i], 
                        action=action[i], 
                        reward=reward[i], 
                        discount=discount[i], 
                        reset=reset[i], 
                        mu_prob=mu_prob[i], 
                        mu=mu[i] if mu is not None else mu, 
                        mu_mean=mu_mean[i] if mu_mean is not None else mu_mean, 
                        mu_std=mu_std[i] if mu_std is not None else mu_std, 
                        state=self.model.state_type(*[s[i] for s in state]) if state is not None else state, 
                        mask=mask[i] if mask is not None else mask, 
                        use_meta=True, 
                        debug=True
                    )

                terms[f'grads_norm{i}'], var_norms = self.optimizer(
                    tape, loss, return_var_norms=True)
                terms['var_norm'] = list(var_norms.values())
                grads = self.optimizer.get_transformed_grads()
                grads = list(grads.values())
                terms[f'trans_grads_norm{i}'] = tf.linalg.global_norm(grads)
                grads_list.append(grads)

            meta_loss, meta_terms = self.loss.loss(
                tape=meta_tape, 
                obs=obs[-1], 
                action=action[-1], 
                reward=reward[-1], 
                discount=discount[-1], 
                reset=reset[-1], 
                mu_prob=mu_prob[-1], 
                mu=mu[-1] if mu is not None else mu, 
                mu_mean=mu_mean[-1] if mu_mean is not None else mu_mean, 
                mu_std=mu_std[-1] if mu_std is not None else mu_std, 
                state=self.model.state_type(*[s[-1] for s in state]) if state is not None else state, 
                mask=mask[-1] if mask is not None else mask, 
                name='meta', 
                use_meta=False
            )

        meta_vars = sum([m.variables for m in self.meta_modules], ())
        self.meta_opt.set_variables(meta_vars)

        meta_grads_list = compute_meta_gradients(
            meta_tape=meta_tape, 
            meta_loss=meta_loss, 
            grads_list=grads_list, 
            theta=self.optimizer.variables, 
            eta=meta_vars, 
            inner_steps=self.config.inner_steps
        )
        assert len(meta_grads_list) == self.config.inner_steps, (meta_grads_list, self.config.inner_steps)
        meta_grads = [sum(mg) for mg in zip(*meta_grads_list)]
        assert len(meta_grads) == len(meta_vars), (len(meta_grads), len(meta_vars))
        meta_terms['meta_grads_norm'], clipped_meta_grads = \
            self.meta_opt.apply_gradients(meta_grads, vars=meta_vars, return_grads=True)

        def record_meta_stats(meta_terms):
            for i, mg in enumerate(reversed(meta_grads_list)):
                mg = {v.name: g for v, g in zip(meta_vars, mg)}
                meta_terms = _add_norm(
                    meta_terms, mg, 'meta', f'grads{i}', f'grads_norm{i}')
            mg = {v.name: g for v, g in zip(meta_vars, meta_grads)}
            meta_terms = _add_norm(
                meta_terms, mg, 'meta', 'grads', None
            )
            meta_terms = _add_norm(
                meta_terms, clipped_meta_grads, 'meta', 
                'clipped_grads', 'clipped_grads_norm'
            )
            trans_meta_grads = self.meta_opt.get_transformed_grads()
            meta_terms = _add_norm(
                meta_terms, trans_meta_grads, 'meta',
                'trans_grads', 'trans_grads_norm'
            )
            var_norms = self.meta_opt.get_var_norms()
            meta_terms = _add_norm(
                meta_terms, var_norms, 'meta', 'norm', 'var_norm'
            )
            return meta_terms
        meta_terms = record_meta_stats(meta_terms)
        terms.update(meta_terms)
        return terms

create_trainer = functools.partial(create_trainer,
    name='zero', trainer_cls=Trainer
)
