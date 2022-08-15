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
from .utils import get_hx


def _get_rl_modules(model):
    modules = tuple([
        v for k, v in model.items() if not k.startswith('meta')
    ])
    return modules

def _get_meta_modules(model):
    modules = tuple([
        v for k, v in model.items() if k.startswith('meta') #and k != 'meta'
    ])
    return modules

def _add_norm(terms, d, norm_name=None):
    terms.update({
        k: v for k, v in d.items()
    })
    if norm_name is not None:
        terms[f'{norm_name}'] = tf.linalg.global_norm(list(d.values()))
    return terms


class Trainer(TrainerBase):
    def _add_attributes(self):
        self._use_meta = self.config.K and self.config.L is not None
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
        self.optimizers: Dict[str, Optimizer] = {}
        self.optimizers['rl'] = create_optimizer(
            modules, config.optimizer, f'rl/{opt_name}'
        )
        if self._use_meta:
            modules = _get_rl_modules(self.model['meta'])
            do_logging(modules, prefix='Meta RL Modules', level='print')
            self.optimizers['meta_rl'] = create_optimizer(
                modules, config.optimizer, f'meta_rl/{opt_name}'
            )

            opt_name = config.meta_opt.opt_name
            config.meta_opt.opt_name = opts[opt_name]
            # self.meta_param_module = self.model['meta'].meta
            # do_logging(self.meta_param_module, prefix='Meta Parameter Modules', level='print')
            # self.optimizers['meta'] = create_optimizer(
            #     self.meta_param_module, config.meta_param_opt, f'meta/{opt_name}'
            # )
            self.meta_modules = _get_meta_modules(self.model['meta'])
            do_logging(self.meta_modules, prefix='Meta Modules', level='print')
            self.optimizers['meta'] = create_optimizer(
                self.meta_modules, config.meta_opt, f'meta/{opt_name}'
            )

    def sync_opt_vars(self):
        self.sync_ops.sync_vars(
            self.optimizers['rl'].opt_variables, self.optimizers['meta_rl'].opt_variables)

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

    def _inner_epoch(
        self, 
        *, 
        opt, 
        loss_fn, 
        obs, 
        idx=None, 
        event=None, 
        global_state=None, 
        next_obs=None, 
        next_idx=None, 
        next_event=None, 
        next_global_state=None, 
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
        debug=True, 
        return_grads=False
    ):
        n_mbs = self.config.get('n_mbs', 1)
        grads_list = []
        if n_mbs > 1:
            indices = tf.range(obs.shape[0], dtype=tf.int32)
            indices = tf.random.shuffle(indices)
            indices = tf.reshape(indices, (n_mbs, -1))
            for i in range(n_mbs):
                k = indices[i]
                with tf.GradientTape() as tape:
                    loss, terms = loss_fn(
                        tape=tape, 
                        obs=tf.gather(obs, k), 
                        idx=idx if idx is None else tf.gather(idx, k), 
                        event=event if event is None else tf.gather(event, k), 
                        global_state=global_state if global_state is None else tf.gather(global_state, k), 
                        next_obs=next_obs if next_obs is None else tf.gather(next_obs, k), 
                        next_idx=next_idx if next_idx is None else tf.gather(next_idx, k), 
                        next_event=next_event if next_event is None else tf.gather(next_event, k), 
                        next_global_state=next_global_state if next_global_state is None else tf.gather(next_global_state, k), 
                        action=tf.gather(action, k), 
                        old_value=tf.gather(value, k), 
                        reward=tf.gather(reward, k), 
                        discount=tf.gather(discount, k), 
                        reset=reset if reset is None else tf.gather(reset, k), 
                        mu_logprob=mu_logprob if mu_logprob is None else tf.gather(mu_logprob, k), 
                        mu=mu if mu is None else tf.gather(mu, k), 
                        mu_mean=mu_mean if mu_mean is None else tf.gather(mu_mean, k), 
                        mu_std=mu_std if mu_std is None else tf.gather(mu_std, k), 
                        action_mask=action_mask if action_mask is None else tf.gather(action_mask, k), 
                        sample_mask=life_mask if life_mask is None else tf.gather(life_mask, k), 
                        prev_reward=prev_reward if prev_reward is None else tf.gather(prev_reward, k), 
                        prev_action=prev_action if prev_action is None else tf.gather(prev_action, k), 
                        state=state, 
                        mask=mask, 
                        use_meta=use_meta, 
                        debug=debug
                    )
                terms['grads_norm'], var_norms = opt(
                    tape, loss, return_var_norms=True
                )
                terms['var_norm'] = list(var_norms.values())
                if return_grads:
                    grads = opt.get_transformed_grads()
                    grads = list(grads.values())
                    for g in grads:
                        tf.debugging.assert_all_finite(g, f'Bad {g.name}')
                    terms[f'trans_grads_norm'] = tf.linalg.global_norm(grads)
                    grads_list.append(grads)
        else:
            with tf.GradientTape() as tape:
                loss, terms = loss_fn(
                    tape=tape, 
                    obs=obs, 
                    idx=idx, 
                    event=event, 
                    global_state=global_state, 
                    next_obs=next_obs, 
                    next_idx=next_idx, 
                    next_event=next_event, 
                    next_global_state=next_global_state, 
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
                    debug=debug
                )
                terms['grads_norm'], var_norms = opt(
                    tape, loss, return_var_norms=True
                )
                terms['var_norm'] = list(var_norms.values())
                if return_grads:
                    grads = opt.get_transformed_grads()
                    grads = list(grads.values())
                    for g in grads:
                        tf.debugging.assert_all_finite(g, f'Bad {g.name}')
                    terms['trans_grads_norm'] = tf.linalg.global_norm(grads)
                    grads_list.append(grads)

        if return_grads:
            return terms, grads_list
        else:
            return terms

    def _outer_grads(
        self, 
        *, 
        tape, 
        grads_list, 
        obs, 
        idx=None, 
        event=None, 
        hidden_state=None, 
        next_obs=None, 
        next_idx=None, 
        next_event=None, 
        next_hidden_state=None, 
        action, 
        value, 
        meta_reward, 
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
        meta_loss, terms = self.loss.meta.outer_loss(
            tape=tape, 
            obs=obs, 
            idx=idx, 
            event=event, 
            hidden_state=hidden_state, 
            next_obs=next_obs, 
            next_idx=next_idx, 
            next_event=next_event, 
            next_hidden_state=next_hidden_state, 
            action=action, 
            old_value=value, 
            meta_reward=meta_reward, 
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
            name='meta', 
            use_meta=False
        )
        with tape.stop_recording():
            meta_vars = sum([m.variables for m in self.meta_modules], ())
            self.optimizers['meta'].set_variables(meta_vars)
            meta_grads_list = compute_meta_gradients(
                meta_tape=tape, 
                meta_loss=meta_loss, 
                grads_list=grads_list, 
                theta=self.optimizers['meta_rl'].variables, 
                eta=meta_vars, 
            )
            meta_grads_tensors = [tf.stack(mg) for mg in meta_grads_list]
            meta_grads = [sum(mg) for mg in meta_grads_list]
            for v, g in zip(meta_vars, meta_grads_tensors):
                terms[f'{v.name.split(":")[0]}:step_grads'] = g
            assert len(meta_grads) == len(meta_vars), (len(meta_grads), len(meta_vars))
        
        return meta_grads, meta_vars, terms

    def _apply_meta_grads(self, meta_grads, meta_vars, terms):
        terms['meta/grads_norm'], clipped_meta_grads = \
            self.optimizers['meta'].apply_gradients(
                meta_grads, return_grads=True)
        mg = {f'{v.name.split(":")[0]}/grads': g for v, g in zip(meta_vars, meta_grads)}
        terms = _add_norm(terms, mg)
        mv = {f'{v.name.split(":")[0]}/var': v for v in meta_vars}
        terms = _add_norm(terms, mv)
        cmg = {f'{k.split(":")[0]}/clipped_grads': v for k, v in clipped_meta_grads.items()}
        terms = _add_norm(
            terms, cmg, 'meta/clipped_grads_norm'
        )
        trans_meta_grads = self.optimizers['meta'].get_transformed_grads()
        trans_meta_grads = {f'{k.split(":")[0]}/trans_grads': v for k, v in trans_meta_grads.items()}
        terms = _add_norm(
            terms, trans_meta_grads, 'meta/trans_grads_norm'
        )
        var_norms = self.optimizers['meta'].get_var_norms()
        var_norms = {f'{k.split(":")[0]}/var_norm': v for k, v in var_norms.items()}
        terms = _add_norm(terms, var_norms)
        terms['meta/var_norm'] = list(var_norms.values())
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
        life_mask=None, 
        prev_reward=None,
        prev_action=None,
        state=None, 
        mask=None, 
        use_meta=False, 
    ):
        if next_hidden_state is None:
            curr_hidden_state = hidden_state[:, :-1]
            curr_idx = idx[:, :-1]
            curr_event = None if event is None else event[:, :-1]
            if action_mask is not None:
                action_mask = action_mask[:, :-1]
            if life_mask is not None:
                life_mask = life_mask[:, :-1]
        else:
            curr_hidden_state = hidden_state
            curr_idx = idx
            curr_event = event
        _, rl_reward = self._compute_rl_reward(
            curr_hidden_state, 
            action, 
            curr_idx, 
            curr_event, 
            reward
        )
        for _ in range(self.config.n_epochs):
            terms = self._inner_epoch(
                opt=self.optimizers['rl'], 
                loss_fn=self.loss.rl.loss, 
                obs=obs, 
                idx=idx, 
                event=event, 
                global_state=global_state, 
                next_obs=next_obs, 
                next_idx=next_idx, 
                next_event=next_event, 
                next_global_state=next_global_state, 
                action=action, 
                value=value, 
                reward=rl_reward, 
                discount=discount, 
                reset=reset, 
                mu_logprob=mu_logprob, 
                mu=mu, 
                mu_mean=mu_mean, 
                mu_std=mu_std, 
                action_mask=action_mask, 
                life_mask=life_mask, 
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
        life_mask=None, 
        prev_reward=None,
        prev_action=None,
    ):
        inner_steps = self.config.K
        if next_hidden_state is None:
            curr_hidden_state = hidden_state[:, :, :-1]
            curr_idx = idx[:, :, :-1]
            curr_event = None if event is None else event[:, :, :-1]
            if action_mask is not None:
                action_mask = action_mask[:, :, :-1]
            if life_mask is not None:
                life_mask = life_mask[:, :, :-1]
        else:
            curr_hidden_state = hidden_state
            curr_idx = idx
            curr_event = event
        with tf.GradientTape(persistent=True) as meta_tape:
            meta_reward, rl_reward = self._compute_rl_reward(
                curr_hidden_state, 
                action, 
                curr_idx, 
                curr_event, 
                reward
            )
            if event is not None and self.config.event_done:
                event_idx = tf.argmax(event, -1)
                in_discount = tf.cast(event_idx[:, :, :-1] == event_idx[:, :, 1:], tf.float32)
            else:
                in_discount = discount
            for i in range(inner_steps):
                meta_grads = []
                grads_list = []
                for j in range(self.config.n_epochs):
                    terms, gs = self._inner_epoch(
                        opt=self.optimizers['meta_rl'], 
                        loss_fn=self.loss.meta.loss, 
                        obs=obs[i], 
                        idx=None if idx is None else idx[i], 
                        event=None if event is None else event[i], 
                        global_state=None if global_state is None else global_state[i], 
                        next_obs=None if next_obs is None else next_obs[i],
                        next_idx=None if next_idx is None else next_idx[i],
                        next_event=None if next_event is None else next_event[i],
                        next_global_state=None if next_global_state is None else next_global_state[i], 
                        action=action[i], 
                        value=value[i], 
                        reward=rl_reward[i], 
                        discount=in_discount[i], 
                        reset=reset[i], 
                        mu_logprob=mu_logprob[i], 
                        mu=mu[i] if mu is not None else mu, 
                        mu_mean=mu_mean[i] if mu_mean is not None else mu_mean, 
                        mu_std=mu_std[i] if mu_std is not None else mu_std, 
                        action_mask=action_mask[i] if action_mask is not None else action_mask, 
                        life_mask=life_mask[i] if life_mask is not None else life_mask, 
                        prev_reward=prev_reward[i] if prev_reward is not None else prev_reward, 
                        prev_action=prev_action[i] if prev_action is not None else prev_action, 
                        state=self.model.state_type(*[s[i] for s in state]) if state is not None else state, 
                        mask=mask[i] if mask is not None else mask, 
                        use_meta=True, 
                        return_grads=True
                    )
                    grads_list += gs

                    mgs, meta_vars, meta_terms = self._outer_grads(
                        tape=meta_tape, 
                        grads_list=grads_list, 
                        obs=obs[-1], 
                        idx=None if idx is None else idx[-1], 
                        event=None if event is None else event[-1], 
                        hidden_state=None if hidden_state is None else hidden_state[-1], 
                        next_obs=None if next_obs is None else next_obs[-1],
                        next_idx=None if next_idx is None else next_idx[-1],
                        next_event=None if next_event is None else next_event[-1],
                        next_hidden_state= None if next_hidden_state is None else next_hidden_state[-1], 
                        action=action[-1], 
                        value=value[-1], 
                        meta_reward=meta_reward, 
                        reward=reward[-1], 
                        discount=discount[-1], 
                        reset=reset[-1], 
                        mu_logprob=mu_logprob[-1], 
                        mu=mu[-1] if mu is not None else mu, 
                        mu_mean=mu_mean[-1] if mu_mean is not None else mu_mean, 
                        mu_std=mu_std[-1] if mu_std is not None else mu_std, 
                        action_mask=action_mask[-1] if action_mask is not None else action_mask, 
                        life_mask=life_mask[-1] if life_mask is not None else life_mask, 
                        prev_reward=prev_reward[-1] if prev_reward is not None else prev_reward, 
                        prev_action=prev_action[-1] if prev_action is not None else prev_action, 
                        state=self.model.state_type(*[s[-1] for s in state]) if state is not None else state, 
                        mask=mask[-1] if mask is not None else mask, 
                    )
                    meta_grads.append(mgs)
            meta_grads = [sum(mg) / len(mg) for mg in zip(*meta_grads)]
            meta_terms = self._apply_meta_grads(meta_grads, meta_vars, meta_terms)
        terms['meta_reward'] = meta_reward
        terms['rl_reward'] = meta_reward
        terms['reward_coef'] = self.model['meta'].meta('reward_coef', inner=True)
        # if event is not None:
        #     fake_event = tf.one_hot(tf.convert_to_tensor([[0, 1], [1, 0]]), 2)
        #     fake_idx = tf.one_hot(tf.convert_to_tensor([[0, 1], [0, 1]]), 2)
        #     hx = get_hx(fake_idx, fake_event)
        # else:
        #     hx = tf.one_hot(tf.convert_to_tensor([[0, 1], [0, 1]]), 2)
        # fake_obs = hidden_state[0, 0, :2]
        # fake_act = tf.one_hot(action[0, 0, :2], self.model['rl'].policy.action_dim)
        # x = tf.concat([fake_obs, fake_act], -1)
        # fake_in_reward = self.model['meta'].meta_reward(x, hx=hx)
        # terms['in_reward11'] = fake_in_reward[0, 0]
        # terms['in_reward21'] = fake_in_reward[1, 0]
        # if event is not None:
        #     terms['in_reward12'] = fake_in_reward[0, 1]
        #     terms['in_reward22'] = fake_in_reward[1, 1]

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
            self.optimizers[k].set_weights(v)

    def _compute_rl_reward(self, hidden_state, action, idx, event, reward):
        if self.config.K:
            action_oh = tf.one_hot(action, self.model['meta'].policy.action_dim)
            x = tf.concat([hidden_state, action_oh], -1)
            hx = get_hx(idx, event)
            meta_reward = self.model['meta'].meta_reward(x, hx=hx)
            reward_scale = self.model['meta'].meta('reward_scale', inner=True)
            reward_bias = self.model['meta'].meta('reward_bias', inner=True)
            reward = reward_scale * reward + reward_bias
            reward_coef = self.model['meta'].meta('reward_coef', inner=True)
            rl_reward = reward_coef * reward + (1-reward_coef) * meta_reward
            return meta_reward, rl_reward
        else:
            return None, reward

create_trainer = functools.partial(create_trainer,
    name='zero', trainer_cls=Trainer
)
