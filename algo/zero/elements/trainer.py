import functools
from typing import Dict
import tensorflow as tf

from core.elements.trainer import Trainer as TrainerBase, create_trainer
from core.decorator import override
from core.log import do_logging
from core.optimizer import create_optimizer, Optimizer
from core.tf_config import build
from core.utils import get_vars_for_modules
from tools import pkg
from optimizers.adam import Adam
from optimizers.rmsprop import RMSprop
from optimizers.sgd import SGD
from tools.meta import compute_meta_gradients, inner_epoch
from core.typing import AttrDict
from tools.utils import dict2AttrDict
from tools import tf_utils
from .utils import compute_inner_steps, get_rl_modules, \
    get_meta_param_modules


def _add_norm(terms, d, norm_name=None):
    terms.update({
        k: v for k, v in d.items()
    })
    if norm_name is not None:
        terms[f'{norm_name}'] = tf.linalg.global_norm(list(d.values()))
    return terms


class Trainer(TrainerBase):
    def _add_attributes(self):
        self.config = compute_inner_steps(self.config)
        self._use_meta = self.config.inner_steps is not None
        assert self.config.msmg_type in ('avg', 'last'), self.config.msmg_type

    def construct_optimizers(self):
        config = dict2AttrDict(self.config, to_copy=True)
        opts = {
            'adam': Adam, 
            'rmsprop': RMSprop, 
            'sgd': SGD
        }
        opt_name = config.rl_opt.opt_name
        config.rl_opt.opt_name = opts[opt_name]
        rl_modules = get_rl_modules(self.model.rl)
        do_logging(rl_modules, prefix='RL Modules', level='print')
        self.optimizers: Dict[str, Optimizer] = AttrDict()
        self.optimizers.rl = create_optimizer(
            rl_modules, config.rl_opt, f'rl/{opt_name}'
        )
        if self._use_meta:
            rl_modules = get_rl_modules(self.model.meta)
            self.optimizers.meta_rl = create_optimizer(
                rl_modules, config.rl_opt, f'meta_rl/{opt_name}'
            )
            self.meta_param_modules = get_meta_param_modules(self.model.meta)
            opt_name = config.meta_param_opt.opt_name
            config.meta_param_opt.opt_name = opts[opt_name]
            do_logging(self.meta_param_modules, prefix='Meta Param Modules', level='print')
            self.optimizers.meta_param = create_optimizer(
                self.meta_param_modules, config.meta_param_opt, f'meta_param/{opt_name}'
            )

    @tf.function
    def sync_nets(self):
        forward = self.config.extra_meta_step == 0
        if self.config.inner_steps is None or self.config.inner_steps + self.config.extra_meta_step == 1:
            pass
        elif forward:
            self.sync_ops.sync_vars(
                self.optimizers.meta_rl.variables, self.optimizers.rl.variables)
            self.sync_ops.sync_vars(
                self.optimizers.meta_rl.opt_variables, self.optimizers.rl.opt_variables)
        else:
            self.sync_ops.sync_vars(
                self.optimizers.rl.variables, self.optimizers.meta_rl.variables)
            self.sync_ops.sync_vars(
                self.optimizers.rl.opt_variables, self.optimizers.meta_rl.opt_variables)

    def get_rl_weights(self):
        weights = {
            'model': self.model.get_weights(self.rl_names), 
            'opt': self.optimizers.rl.get_weights()
        }
        return weights
    
    def set_rl_weights(self, weights):
        if weights is None:
            return
        self.model.set_weights(weights['model'])
        self.optimizers.rl.set_weights(weights['opt'])

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

    def _outer_grads(
        self, 
        *, 
        tape, 
        grads_list, 
        iteration, 
        **data
    ):
        meta_loss, ml_terms = self.loss.meta.outer_loss(
            tape=tape, name='meta', **data
        )

        with tape.stop_recording():
            mpgs, mpg_terms = self._meta_grads_for_modules(
                meta_tape=tape, 
                meta_modules=self.meta_param_modules, 
                meta_loss=meta_loss, 
                grads_list=grads_list, 
                iteration=iteration, 
            )
            mrgs, mrg_terms = self._meta_grads_for_modules(
                meta_tape=tape, 
                meta_modules=self.meta_reward_modules, 
                meta_loss=meta_loss, 
                grads_list=grads_list, 
                iteration=iteration, 
            )

            terms = {**ml_terms, **mpg_terms, **mrg_terms}

        return mpgs, mrgs, terms

    def _meta_grads_for_modules(
        self, 
        *, 
        meta_tape, 
        meta_modules, 
        meta_loss, 
        grads_list, 
        iteration, 
    ):
        meta_vars = get_vars_for_modules(meta_modules)
        meta_grads = compute_meta_gradients(
            meta_tape=meta_tape, 
            meta_loss=meta_loss, 
            grads_list=grads_list, 
            theta=self.optimizers.rl.variables, 
            eta=meta_vars, 
        )
        terms = {
            f'{v.name.split(":")[0]}:iter{iteration}:step_grads': g
            for v, g in zip(meta_vars, meta_grads)
        }

        return meta_grads, terms

    def _set_opt_vars(self, opt_name, vars):
        if self.optimizers[opt_name].variables is None:
            self.optimizers[opt_name].set_variables(vars)

    def _apply_meta_grads(self, opt_name, meta_grads, meta_vars, terms):
        terms[f'{opt_name}/grads_norm'], clipped_meta_grads = \
            self.optimizers[opt_name].apply_gradients(
                meta_grads, return_grads=True)

        mg = {f'{v.name.split(":")[0]}/grads': g for v, g in zip(meta_vars, meta_grads)}
        terms = _add_norm(terms, mg)
        mv = {f'{v.name.split(":")[0]}/var': v for v in meta_vars}
        terms = _add_norm(terms, mv)
        cmg = {f'{k.split(":")[0]}/clipped_grads': v for k, v in clipped_meta_grads.items()}
        terms = _add_norm(terms, cmg, f'{opt_name}/clipped_grads_norm')
        trans_meta_grads = self.optimizers[opt_name].get_transformed_grads()
        trans_meta_grads = {f'{k.split(":")[0]}/trans_grads': v for k, v in trans_meta_grads.items()}
        terms = _add_norm(terms, trans_meta_grads, f'{opt_name}/trans_grads_norm')
        var_norms = self.optimizers[opt_name].get_var_norms()
        if var_norms:
            var_norms = {f'{k.split(":")[0]}/var_norm': v for k, v in var_norms.items()}
            terms = _add_norm(terms, var_norms)
            terms[f'{opt_name}/var_norm'] = list(var_norms.values())
        
        return terms

    def raw_train(
        self, 
        *, 
        obs, 
        idx=None, 
        event=None, 
        global_state=None, 
        hidden_state=None, 
        next_obs=None, 
        next_idx=None, 
        next_event=None, 
        next_global_state=None, 
        next_hidden_state=None, 
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
        next_action_mask=None, 
        life_mask=None, 
        next_life_mask=None, 
        prev_reward=None,
        prev_action=None,
        state=None, 
        mask=None, 
        use_meta=False, 
    ):
        (action_mask, life_mask), _ = tf_utils.split_data(
            [action_mask, life_mask], 
            [next_action_mask, next_life_mask], 
            axis=1
        )
        _, _, rl_reward = self._compute_rl_reward(
            hidden_state, 
            next_hidden_state, 
            action, 
            idx, 
            next_idx, 
            event, 
            next_event, 
            reward, 
            axis=1
        )
        rl_discount, rl_reset = self._compute_rl_discount(
            discount, event, next_event, reset
        )
        for _ in range(self.config.n_epochs):
            terms = inner_epoch(
                config=self.config, 
                opt=self.optimizers.rl, 
                loss_fn=self.loss.rl.loss, 
                obs=obs, 
                idx=idx, 
                event=event, 
                global_state=global_state, 
                hidden_state=hidden_state, 
                next_obs=next_obs, 
                next_idx=next_idx, 
                next_event=next_event, 
                next_global_state=next_global_state, 
                next_hidden_state=next_hidden_state, 
                action=action, 
                old_value=value, 
                rl_reward=rl_reward, 
                rl_discount=rl_discount, 
                rl_reset=rl_reset, 
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
                debug=not self._use_meta, 
                use_meta=use_meta, 
            )

        return terms

    def raw_meta_train(
        self, 
        *, 
        obs, 
        idx=None, 
        event=None, 
        global_state=None, 
        hidden_state=None, 
        next_obs=None, 
        next_idx=None, 
        next_event=None, 
        next_global_state=None, 
        next_hidden_state=None, 
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
        next_action_mask=None, 
        life_mask=None, 
        next_life_mask=None, 
        prev_reward=None,
        prev_action=None,
    ):        
        inner_steps = self.config.K
        (action_mask, life_mask), _ = tf_utils.split_data(
            [action_mask, life_mask], 
            [next_action_mask, next_life_mask], 
            axis=2
        )
        data = dict(
            obs=obs, 
            idx=idx, 
            event=event, 
            global_state=global_state, 
            hidden_state=hidden_state, 
            next_obs=next_obs,
            next_idx=next_idx,
            next_event=next_event,
            next_global_state=next_global_state, 
            next_hidden_state=next_hidden_state, 
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
        )
        rl_discount, rl_reset = self._compute_rl_discount(
            discount, event, next_event, reset
        )
        with tf.GradientTape(persistent=True) as meta_tape:
            meta_reward, trans_reward, rl_reward = self._compute_rl_reward(
                hidden_state, 
                next_hidden_state, 
                action, 
                idx, 
                next_idx, 
                event, 
                next_event, 
                reward, 
                axis=2
            )
            data['meta_reward'] = meta_reward
            
            meta_param_grads = []
            grads_list = []
            for i in range(inner_steps):
                rl_data_i = tf_utils.gather(data, i)
                data_i = tf_utils.gather(data, i+self.config.extra_meta_step)
                for j in range(1, self.config.n_epochs+1):
                    terms, gl = inner_epoch(
                        config=self.config, 
                        opt=self.optimizers.meta_rl, 
                        loss_fn=self.loss.meta.loss, 
                        **rl_data_i,
                        use_meta=True, 
                        use_dice=j == 1, 
                        return_grads=True
                    )
                    grads_list += gl

                    if self.config.msmg_type == 'avg':
                        mpgs, meta_terms = self._outer_grads(
                            tape=meta_tape, 
                            grads_list=grads_list, 
                            iteration=i, 
                            **data_i, 
                        )
                        meta_param_grads.append(mpgs)
            if self.config.msmg_type == 'last':
                data_final = tf_utils.gather(data, i+self.config.extra_meta_step)
                for _ in range(self.config.n_epochs):
                    meta_param_grads, meta_terms = self._outer_grads(
                        tape=meta_tape, 
                        grads_list=grads_list, 
                        iteration=inner_steps, 
                        **data_final
                    )
        if self.config.msmg_type == 'avg':
            meta_param_grads = [sum(mg) / len(mg) for mg in zip(*meta_param_grads)]

        meta_param_vars = get_vars_for_modules(self.meta_param_modules)
        meta_reward_vars = get_vars_for_modules(self.meta_reward_modules)
        self._set_opt_vars('meta_param', meta_param_vars)
        self._set_opt_vars('meta_reward', meta_reward_vars)
        meta_terms = self._apply_meta_grads(
            'meta_param', meta_param_grads, meta_param_vars, meta_terms)

        if self.config.extra_meta_step:
            rl_data_final = tf_utils.gather(data, inner_steps)
            for j in range(self.config.n_epochs):
                terms = inner_epoch(
                    config=self.config, 
                    opt=self.optimizers.meta_rl, 
                    loss_fn=self.loss.meta.loss, 
                    **rl_data_final, 
                    use_meta=True, 
                    use_dice=False, 
                    return_grads=False
                )
        terms.update(meta_terms)

        return terms


create_trainer = functools.partial(create_trainer,
    name='zero', trainer_cls=Trainer
)
