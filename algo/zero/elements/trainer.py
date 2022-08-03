import functools
from typing import Dict
import tensorflow as tf

from core.elements.trainer import Trainer as TrainerBase, create_trainer
from core.decorator import override
from core.log import do_logging
from core.optimizer import create_optimizer, Optimizer
from core.tf_config import build
from utility import pkg
from optimizers.adam import Adam
from optimizers.rmsprop import RMSprop
from utility.meta import compute_meta_gradients
from utility.utils import dict2AttrDict


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

def _add_norm(terms, d, norm_name):
    terms.update({
        k: v for k, v in d.items()
    })
    if norm_name is not None:
        terms[f'{norm_name}'] = tf.linalg.global_norm(list(d.values()))
    return terms


class Trainer(TrainerBase):
    def _add_attributes(self):
        self._use_meta = self.config.K is not None and self.config.L is not None
        self.config.inner_steps = self.config.K + self.config.L if self._use_meta else None

    def construct_optimizers(self):
        config = dict2AttrDict(self.config, to_copy=True)
        opts = {
            'adam': Adam, 
            'rmsprop': RMSprop
        }
        opt_name = config.optimizer.opt_name
        config.optimizer.opt_name = opts[opt_name]
        modules = _get_rl_modules(self.model['rl'])
        do_logging(modules, prefix='RL modules', level='print')
        self.optimizers: Dict[Optimizer] = {}
        self.optimizers['rl'] = create_optimizer(
            modules, config.optimizer, f'rl/{opt_name}'
        )
        if self._use_meta:
            modules = _get_rl_modules(self.model['meta'])
            do_logging(modules, prefix='Meta RL modules', level='print')
            self.optimizers['meta_rl'] = create_optimizer(
                modules, config.optimizer, f'meta_rl/{opt_name}'
            )

            opt_name = config.meta_opt.opt_name
            config.meta_opt.opt_name = opts[opt_name]
            self.meta_modules = _get_meta_modules(self.model['meta'])
            do_logging(self.meta_modules, prefix='Meta modules', level='print')
            self.optimizers['meta'] = create_optimizer(
                self.meta_modules, config.meta_opt, f'meta/{opt_name}'
            )

    def sync_opt_vars(self):
        self.sync_ops.sync_vars(
            self.optimizers['rl'].opt_variables, self.optimizers['meta_rl'].opt_variables)

    def sync_meta_vars(self):
        self.sync_ops.sync_nets(
            self.meta_modules, _get_meta_modules(self.model['rl']))

    def sync_nets(self, forward=True):
        if self._use_meta:
            self.sync_opt_vars()
            self.model.sync_nets(forward=forward)

    def ckpt_model(self):
        opts = {
            f'{self._raw_name}_{k}_opt': v
            for k, v in self.optimizers.items()
        }
        return opts

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
        global_state=None, 
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
        action_mask=None, 
        life_mask=None, 
        prev_reward=None,
        prev_action=None,
        state=None, 
        mask=None, 
        use_meta=False, 
    ):
        if action_mask is not None:
            action_mask = action_mask[:, :-1]
        if life_mask is not None:
            life_mask = life_mask[:, :-1]
        for _ in range(self.config.n_epochs):
            n_mbs = self.config.get('n_mbs', 1)
            if n_mbs > 1:
                indices = tf.range(obs.shape[0], dtype=tf.int32)
                indices = tf.random.shuffle(indices)
                indices = tf.reshape(indices, (n_mbs, -1))
                for i in range(n_mbs):
                    idx = indices[i]
                    with tf.GradientTape() as tape:
                        loss, terms = self.loss.rl.loss(
                            tape=tape, 
                            obs=obs if obs is None else tf.gather(obs, idx), 
                            global_state=global_state if global_state is None else tf.gather(global_state, idx), 
                            next_obs=next_obs if next_obs is None else tf.gather(next_obs, idx), 
                            action=action if action is None else tf.gather(action, idx), 
                            old_value=value if value is None else tf.gather(value, idx), 
                            reward=reward if reward is None else tf.gather(reward, idx), 
                            discount=discount if discount is None else tf.gather(discount, idx), 
                            reset=reset if reset is None else tf.gather(reset, idx), 
                            mu_logprob=mu_logprob if mu_logprob is None else tf.gather(mu_logprob, idx), 
                            mu=mu if mu is None else tf.gather(mu, idx), 
                            mu_mean=mu_mean if mu_mean is None else tf.gather(mu_mean, idx), 
                            mu_std=mu_std if mu_std is None else tf.gather(mu_std, idx), 
                            action_mask=action_mask if action_mask is None else tf.gather(action_mask, idx), 
                            sample_mask=life_mask if life_mask is None else tf.gather(life_mask, idx), 
                            prev_reward=prev_reward if prev_reward is None else tf.gather(prev_reward, idx), 
                            prev_action=prev_action if prev_action is None else tf.gather(prev_action, idx), 
                            state=state, 
                            mask=mask, 
                            use_meta=use_meta, 
                            debug=not self._use_meta
                        )
            else:
                with tf.GradientTape() as tape:
                    loss, terms = self.loss.rl.loss(
                        tape=tape, 
                        obs=obs, 
                        global_state=global_state, 
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
                        action_mask=action_mask, 
                        sample_mask=life_mask, 
                        prev_reward=prev_reward, 
                        prev_action=prev_action, 
                        state=state, 
                        mask=mask, 
                        use_meta=use_meta, 
                        debug=not self._use_meta
                    )

            if not self._use_meta:
                terms['grads_norm'], var_norms, grads = self.optimizers['rl'](
                    tape, loss, return_var_norms=True, return_grads=True
                )
                terms['var_norm'] = list(var_norms.values())
                terms['clipped_grads_norm'] = tf.linalg.global_norm(list(grads.values()))
            else:
                terms['grads_norm'] = self.optimizers['rl'](tape, loss)

        return terms

    def raw_meta_train(
        self, 
        *, 
        obs, 
        global_state=None, 
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
        action_mask=None, 
        life_mask=None, 
        prev_reward=None,
        prev_action=None,
    ):
        if action_mask is not None:
            action_mask = action_mask[:, :, :-1]
        if life_mask is not None:
            life_mask = life_mask[:, :, :-1]
        inner_steps = self.config.K
        with tf.GradientTape(persistent=True) as meta_tape:
            grads_list = []
            for i in range(inner_steps):
                for j in range(self.config.n_epochs):
                    action_oh = tf.one_hot(action[i], self.loss.model.action_dim)
                    obs = tf.concat([obs[i], action_oh], -1)
                    in_reward = self.model['meta_reward'](obs[i])
                    with tf.GradientTape() as tape:
                        loss, terms = self.loss.meta.loss(
                            tape=tape, 
                            obs=obs[i], 
                            global_state=global_state[i], 
                            next_obs=None if next_obs is None else next_obs[i],
                            action=action[i], 
                            old_value=value[i], 
                            reward=in_reward, 
                            discount=discount[i], 
                            reset=reset[i], 
                            mu_logprob=mu_logprob[i], 
                            mu=mu[i] if mu is not None else mu, 
                            mu_mean=mu_mean[i] if mu_mean is not None else mu_mean, 
                            mu_std=mu_std[i] if mu_std is not None else mu_std, 
                            action_mask=action_mask[i] if action_mask is not None else action_mask, 
                            sample_mask=life_mask[i] if life_mask is not None else life_mask, 
                            prev_reward=prev_reward[i] if prev_reward is not None else prev_reward, 
                            prev_action=prev_action[i] if prev_action is not None else prev_action, 
                            state=self.model.state_type(*[s[i] for s in state]) if state is not None else state, 
                            mask=mask[i] if mask is not None else mask, 
                            use_meta=True, 
                            debug=True
                        )
                    terms[f'grads_norm'], var_norms = self.optimizers['meta_rl'](
                        tape, loss, return_var_norms=True
                    )
                    terms['var_norm'] = list(var_norms.values())
                    grads = self.optimizers['meta_rl'].get_transformed_grads()
                    grads = list(grads.values())
                    terms[f'trans_grads_norm'] = tf.linalg.global_norm(grads)
                    grads_list.append(grads)

            for _ in range(self.config.n_meta_epochs):
                if self.config.meta_type == 'plain':
                    meta_loss, meta_terms = self.loss.meta.loss(
                        tape=meta_tape, 
                        obs=obs[-1], 
                        global_state=global_state[-1], 
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
                        action_mask=action_mask[-1] if action_mask is not None else action_mask, 
                        sample_mask=life_mask[-1] if life_mask is not None else life_mask, 
                        prev_reward=prev_reward[-1] if prev_reward is not None else prev_reward, 
                        prev_action=prev_action[-1] if prev_action is not None else prev_action, 
                        state=self.model.state_type(*[s[-1] for s in state]) if state is not None else state, 
                        mask=mask[-1] if mask is not None else mask, 
                        name='meta', 
                        use_meta=False
                    )
                elif self.config.meta_type == 'bmg':
                    meta_loss, meta_terms = self.loss.meta.bmg_loss(
                        tape=meta_tape, 
                        obs=obs[-1], 
                        global_state=global_state[-1], 
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
                        action_mask=action_mask[-1], 
                        life_mask=life_mask[-1], 
                        prev_reward=prev_reward[-1], 
                        prev_action=prev_action[-1], 
                        state=self.model.state_type(*[s[-1] for s in state]) if state is not None else state, 
                        mask=mask[-1] if mask is not None else mask, 
                        name='meta', 
                        use_meta=False
                    )
                else:
                    raise NotImplementedError

            meta_vars = sum([m.variables for m in self.meta_modules], ())
            self.optimizers['meta'].set_variables(meta_vars)

            meta_grads_list = compute_meta_gradients(
                meta_tape=meta_tape, 
                meta_loss=meta_loss, 
                grads_list=grads_list, 
                theta=self.optimizers['meta_rl'].variables, 
                eta=meta_vars, 
                inner_steps=inner_steps
            )
            meta_grads_tensors = [tf.stack(mg) for mg in meta_grads_list]
            meta_grads = [sum(mg) for mg in meta_grads_list]
            for v, g in zip(meta_vars, meta_grads_tensors):
                meta_terms[f'{v.name.split(":")[0]}:step_grads'] = g
            assert len(meta_grads) == len(meta_vars), (len(meta_grads), len(meta_vars))
            meta_terms['meta/grads_norm'], clipped_meta_grads = \
                self.optimizers['meta'].apply_gradients(meta_grads, vars=meta_vars, return_grads=True)

            def record_meta_stats(meta_terms):
                for v, mg in zip(meta_vars, meta_grads_list):
                    mg = {f'{v.name.split(":")[0]}/step_grads{i}': g for i, g in enumerate(mg)}
                    meta_terms = _add_norm(
                        meta_terms, mg, f'meta/grads_norm{i}'
                    )
                mg = {f'{v.name.split(":")[0]}/grads': g for v, g in zip(meta_vars, meta_grads)}
                meta_terms = _add_norm(
                    meta_terms, mg, None
                )
                mv = {f'{v.name.split(":")[0]}/var': v for v in meta_vars}
                meta_terms = _add_norm(
                    meta_terms, mv, None
                )
                meta_terms['meta/var'] = meta_vars
                cmg = {f'{k.split(":")[0]}/clipped_grads': v for k, v in clipped_meta_grads.items()}
                meta_terms = _add_norm(
                    meta_terms, cmg, 'meta/clipped_grads_norm'
                )
                trans_meta_grads = self.optimizers['meta'].get_transformed_grads()
                trans_meta_grads = {f'{k.split(":")[0]}/trans_grads': v for k, v in trans_meta_grads.items()}
                meta_terms = _add_norm(
                    meta_terms, trans_meta_grads, 'meta/trans_grads_norm'
                )
                var_norms = self.optimizers['meta'].get_var_norms()
                var_norms = {f'{k.split(":")[0]}/var_norm': v for k, v in var_norms.items()}
                meta_terms = _add_norm(
                    meta_terms, var_norms, None
                )
                meta_terms['meta/var_norm'] = list(var_norms.values())
                return meta_terms
            meta_terms = record_meta_stats(meta_terms)
            terms.update(meta_terms)

        return terms

    def get_optimizer_weights(self):
        weights = {
            k: v.get_weights()
            for k, v in self.optimizers.items()
        }
        return weights

    def set_optimizer_weights(self, weights):
        for k, v in weights.items():
            self.optimizers[k] = v


create_trainer = functools.partial(create_trainer,
    name='zero', trainer_cls=Trainer
)
