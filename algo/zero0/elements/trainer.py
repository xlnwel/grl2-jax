import functools
import tensorflow as tf

from core.elements.trainer import Trainer as TrainerBase, create_trainer
from core.decorator import override
from core.log import do_logging
from core.optimizer import create_optimizer
from core.tf_config import build
from tools import pkg
from optimizers.adam import Adam
from optimizers.rmsprop import RMSprop
from tools.meta import compute_meta_gradients
from tools.utils import dict2AttrDict


def _get_rl_modules(model):
    modules = tuple([
        v for k, v in model.items() if not k.startswith('meta')
    ])
    return modules

def _get_meta_modules(model):
    modules = tuple([
        v for k, v in model.items() if k.startswith('meta')
    ])
    return modules

def _add_norm(terms, d, tag, suffix, norm_name):
    # terms.update({
    #     f'{k.split(":")[0]}:{suffix}': v for k, v in d.items()
    # })
    if norm_name is not None:
        terms[f'{tag}/{norm_name}'] = tf.linalg.global_norm(list(d.values()))
    return terms


class Trainer(TrainerBase):
    def _add_attributes(self):
        self._use_meta = self.config.K is not None and self.config.L is not None
        self.config.inner_steps = self.config.K + self.config.L if self._use_meta else None

    def construct_optimizers(self):
        if self._use_meta:
            config = dict2AttrDict(self.config, to_copy=True)
            opts = {
                'adam': Adam, 
                'rmsprop': RMSprop
            }
            opt_name = config.optimizer.opt_name
            config.optimizer.opt_name = opts[opt_name]
            modules = _get_rl_modules(self.model['rl'])
            do_logging(modules, prefix='RL modules', level='print')
            self.rl_opt = create_optimizer(
                modules, config.optimizer, f'rl/{opt_name}'
            )
            modules = _get_rl_modules(self.model['meta'])
            do_logging(modules, prefix='Meta RL modules', level='print')
            self.meta_rl_opt = create_optimizer(
                modules, config.optimizer, f'meta_rl/{opt_name}'
            )

            opt_name = config.meta_opt.opt_name
            config.meta_opt.opt_name = opts[opt_name]
            self.meta_modules = _get_meta_modules(self.model['meta'])
            do_logging(self.meta_modules, prefix='Meta modules', level='print')
            self.meta_opt = create_optimizer(
                self.meta_modules, config.meta_opt, f'meta/{opt_name}'
            )
        else:
            keys = sorted([
                k for k in self.model['rl'].keys() if not k.startswith('target')])
            modules = tuple(self.model['rl'][k] for k in keys)
            self.rl_opt = create_optimizer(
                modules, self.config.optimizer)

    def sync_opt_vars(self):
        self.sync_ops.sync_vars(
            self.rl_opt.opt_variables, self.meta_rl_opt.opt_variables)

    def sync_meta_vars(self):
        self.sync_ops.sync_nets(
            self.meta_modules, _get_meta_modules(self.model['rl']))

    def sync_nets(self):
        if self._use_meta:
            self.sync_opt_vars()
            self.model.sync_nets()
            rl_vars = self.rl_opt.opt_variables
            meta_vars = self.meta_rl_opt.opt_variables
            for v1, v2 in zip(rl_vars, meta_vars):
                tf.debugging.assert_equal(v1, v2, message=v1.name)
            rl_vars = self.model['rl'].variables
            meta_vars = self.model['meta'].variables
            assert rl_vars
            assert meta_vars
            for v1, v2 in zip(rl_vars, meta_vars):
                tf.debugging.assert_equal(v1, v2, message=v1.name)

    def ckpt_model(self):
        if self._use_meta:
            models = {
                f'{self._raw_name}_opt': self.rl_opt, 
                f'{self._raw_name}_meta_rl_opt': self.meta_rl_opt, 
                f'{self._raw_name}_meta_opt': self.meta_opt, 
            }
        else:
            models = {
                f'{self._raw_name}_opt': self.rl_opt, 
            }
        return models

    @override(TrainerBase)
    def _build_train(self, env_stats):
        algo = self.config.algorithm.split('-')[-1]
        get_data_format = pkg.import_module(
            'elements.utils', algo=algo).get_data_format
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = get_data_format(
            self.config, env_stats, self.loss.model, meta=False)
        TensorSpecs['use_meta'] = ((), tf.bool, 'use_meta')
        do_logging(TensorSpecs, prefix='Tensor Specifications', level='print')
        self.train = build(self.train, TensorSpecs)
        if self._use_meta:
            meta_train = tf.function(self.raw_meta_train)
            TensorSpecs = get_data_format(
                self.config, env_stats, self.loss.model, meta=True)
            do_logging(TensorSpecs, prefix='Meta Tensor Specifications', level='print')
            self.meta_train = build(meta_train, TensorSpecs)
        return True

    def raw_train(
        self, 
        *, 
        obs, 
        next_obs=None, 
        action, 
        value, 
        reward, 
        discount, 
        reset, 
        mu_logprob, 
        mu=None, 
        mu_mean=None, 
        mu_std=None, 
        state=None, 
        mask=None, 
        use_meta=False, 
    ):
        with tf.GradientTape() as tape:
            loss, terms = self.loss.rl.loss(
                tape=tape, 
                obs=obs, 
                next_obs=next_obs, 
                action=action, 
                old_value=value, 
                reward=reward, 
                discount=discount, 
                reset=reset, 
                mu_logprob=mu_logprob, 
                mu=mu, 
                mu_mean=mu_mean, 
                mu_std=mu_std, 
                state=state, 
                mask=mask, 
                use_meta=use_meta, 
                debug=not self._use_meta
            )

        if not self._use_meta:
            terms['grads_norm'], var_norms, grads = self.rl_opt(
                tape, loss, return_var_norms=True, return_grads=True
            )
            terms['var_norm'] = list(var_norms.values())
            terms['clipped_grads_norm'] = tf.linalg.global_norm(list(grads.values()))
        else:
            terms['grads_norm'] = self.rl_opt(tape, loss)

        return terms

    def raw_meta_train(
        self, 
        *, 
        obs, 
        next_obs=None, 
        action, 
        value, 
        reward, 
        discount, 
        reset, 
        mu_logprob, 
        mu=None, 
        mu_mean=None, 
        mu_std=None, 
        state=None, 
        mask=None
    ):
        inner_steps = self.config.K
        with tf.GradientTape(persistent=True) as meta_tape:
            grads_list = []
            for i in range(inner_steps):
                with tf.GradientTape() as tape:
                    loss, terms = self.loss.meta.loss(
                        tape=tape, 
                        obs=obs[i], 
                        next_obs=None if next_obs is None else next_obs[i],
                        action=action[i], 
                        old_value=value[i], 
                        reward=reward[i], 
                        discount=discount[i], 
                        reset=reset[i], 
                        mu_logprob=mu_logprob[i], 
                        mu=mu[i] if mu is not None else mu, 
                        mu_mean=mu_mean[i] if mu_mean is not None else mu_mean, 
                        mu_std=mu_std[i] if mu_std is not None else mu_std, 
                        state=self.model.state_type(*[s[i] for s in state]) if state is not None else state, 
                        mask=mask[i] if mask is not None else mask, 
                        use_meta=True, 
                        debug=True
                    )

                terms[f'grads_norm'], var_norms = self.meta_rl_opt(
                    tape, loss, return_var_norms=True
                )
                terms['var_norm'] = list(var_norms.values())
                grads = self.meta_rl_opt.get_transformed_grads()
                grads = list(grads.values())
                terms[f'trans_grads_norm'] = tf.linalg.global_norm(grads)
                grads_list.append(grads)

            if self.config.meta_type == 'plain':
                meta_loss, meta_terms = self.loss.meta.loss(
                    tape=meta_tape, 
                    obs=obs[-1], 
                    next_obs=None if next_obs is None else next_obs[-1],
                    action=action[-1], 
                    old_value=value[-1], 
                    reward=reward[-1], 
                    discount=discount[-1], 
                    reset=reset[-1], 
                    mu_logprob=mu_logprob[-1], 
                    mu=mu[-1] if mu is not None else mu, 
                    mu_mean=mu_mean[-1] if mu_mean is not None else mu_mean, 
                    mu_std=mu_std[-1] if mu_std is not None else mu_std, 
                    state=self.model.state_type(*[s[-1] for s in state]) if state is not None else state, 
                    mask=mask[-1] if mask is not None else mask, 
                    name='meta', 
                    use_meta=False
                )
            elif self.config.meta_type == 'bmg':
                meta_loss, meta_terms = self.loss.meta.bmg_loss(
                    tape=meta_tape, 
                    obs=obs[-1], 
                    next_obs=None if next_obs is None else next_obs[-1],
                    action=action[-1], 
                    old_value=value[-1], 
                    reward=reward[-1], 
                    discount=discount[-1], 
                    reset=reset[-1], 
                    mu_logprob=mu_logprob[-1], 
                    mu=mu[-1] if mu is not None else mu, 
                    mu_mean=mu_mean[-1] if mu_mean is not None else mu_mean, 
                    mu_std=mu_std[-1] if mu_std is not None else mu_std, 
                    state=self.model.state_type(*[s[-1] for s in state]) if state is not None else state, 
                    mask=mask[-1] if mask is not None else mask, 
                    name='meta', 
                    use_meta=False
                )
            else:
                raise NotImplementedError

        meta_vars = sum([m.variables for m in self.meta_modules], ())
        self.meta_opt.set_variables(meta_vars)

        meta_grads_list = compute_meta_gradients(
            meta_tape=meta_tape, 
            meta_loss=meta_loss, 
            grads_list=grads_list, 
            theta=self.meta_rl_opt.variables, 
            eta=meta_vars, 
            inner_steps=inner_steps
        )
        assert len(meta_grads_list) == inner_steps+1, (meta_grads_list, inner_steps)
        meta_grads = [sum(mg) for mg in zip(*meta_grads_list)]
        meta_grads_tensors = [tf.stack(mg) for mg in zip(*meta_grads_list)]
        for v, g in zip(meta_vars, meta_grads_tensors):
            meta_terms[f'{v.name.split(":")[0]}:step_grads_std'] = tf.reduce_mean(tf.math.reduce_std(g, 0))
        assert len(meta_grads) == len(meta_vars), (len(meta_grads), len(meta_vars))
        if self.config.use_meta_grads:
            meta_terms['meta/grads_norm'], clipped_meta_grads = \
                self.meta_opt.apply_gradients(meta_grads, vars=meta_vars, return_grads=True)
        else:
            meta_terms['meta/grads_norm'], clipped_meta_grads = self.meta_opt(
                meta_tape, meta_loss, return_var_norms=True
            )

        def record_meta_stats(meta_terms):
            for i, mg in enumerate(reversed(meta_grads_list)):
                mg = {v.name: g for v, g in zip(meta_vars, mg)}
                meta_terms = _add_norm(
                    meta_terms, mg, 'meta', f'grads{i}', f'grads_norm{i}')
            mg = {v.name: g for v, g in zip(meta_vars, meta_grads)}
            meta_terms = _add_norm(
                meta_terms, mg, 'meta', 'grads', None
            )
            mv = {v.name: v for v in meta_vars}
            meta_terms = _add_norm(
                meta_terms, mv, 'meta', 'var', None
            )
            # meta_terms['meta/var'] = meta_vars
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
            # meta_terms = _add_norm(
            #     meta_terms, var_norms, 'meta', 'norm', None
            # )
            meta_terms['meta/var_norm'] = list(var_norms.values())
            return meta_terms
        meta_terms = record_meta_stats(meta_terms)
        terms.update(meta_terms)

        return terms

create_trainer = functools.partial(create_trainer,
    name='zero', trainer_cls=Trainer
)
