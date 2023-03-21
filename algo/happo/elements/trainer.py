from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

from core.ckpt.pickle import save, restore
from core.log import do_logging
from core.elements.trainer import TrainerBase, create_trainer
from core import optimizer
from core.typing import AttrDict
from tools.display import print_dict_info
from tools.rms import RunningMeanStd
from tools.timer import Timer
from tools.utils import flatten_dict, prefix_name, batch_dicts
from algo.lka_common.elements.model import LOOKAHEAD, pop_lookahead
from algo.lka_common.elements.trainer import *


class Trainer(TrainerBase):
    def add_attributes(self):
        self.popart = [RunningMeanStd((0, 1)) for _ in self.model.aid2uids]
        self.indices = np.arange(self.config.n_runners * self.config.n_envs)
        self.n_agents = self.env_stats.n_agents
        self.aid2uids = self.env_stats.aid2uids

    def build_optimizers(self):
        theta = self.model.theta.copy()
        theta_policies, _ = pop_lookahead(theta.policies)
        if self.config.get('theta_opt'):
            self.opts.theta, self.params.theta = [list(x)
                for x in zip(*[optimizer.build_optimizer(
                params=AttrDict(policy=p, value=v), 
                **self.config.theta_opt, 
                name=f'theta{i}'
            ) for i, (p, v) in enumerate(zip(theta_policies, theta.vs))])]
        else:
            self.params.theta = AttrDict()
            self.opts.policies, self.params.theta.policies = [list(x)
                for x in zip(*[optimizer.build_optimizer(
                params=p, 
                **self.config.policy_opt, 
                name=f'policy{i}'
            ) for i, p in enumerate(theta_policies)])]
            self.opts.vs, self.params.theta.vs = [list(x)
                for x in zip(*[optimizer.build_optimizer(
                params=v, 
                **self.config.value_opt, 
                name=f'value{i}'
            ) for i, v in enumerate(theta.vs)])]
        self.lookahead_opt_state = self.params.theta

    def compile_train(self):
        _jit_train = jax.jit(self.theta_train, 
            static_argnames=['aid', 'compute_teammate_log_ratio'])
        def jit_train(*args, **kwargs):
            self.rng, rng = jax.random.split(self.rng)
            return _jit_train(*args, rng=rng, **kwargs)
        self.jit_train = jit_train
        self.jit_lka_train = jit_train

        self.haiku_tabulate()

    def train(self, data: AttrDict):
        if self.config.n_runners * self.config.n_envs < self.config.n_mbs:
            self.indices = np.arange(self.config.n_mbs)
            data = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (self.config.n_mbs, -1, *x.shape[2:])), data)

        theta = self.model.theta.copy()
        theta.policies, is_lookahead = pop_lookahead(theta.policies)
        assert all([lka == False for lka in is_lookahead]), is_lookahead
        opt_state = self.params.theta

        if self.config.update_scheme == 'step':
            theta, opt_state, stats = self.stepwise_sequential_opt(
                theta, opt_state, data, self.config.n_epochs, 
                self.config.n_mbs, self.indices, self.jit_train
            )
        else:
            theta, opt_state, stats = self.sequential_opt(
                theta, opt_state, data, self.config.n_epochs, 
                self.config.n_mbs, self.indices, self.jit_train
            )

        for p in theta.policies:
            p[LOOKAHEAD] = False
        self.model.params = theta
        self.params.theta = opt_state

        if self.config.popart:
            for aid, uids in enumerate(self.env_stats.aid2uids):
                self.popart[aid].update(stats.v_target[:, :, uids])

        data = flatten_dict({k: v 
            for k, v in data.items() if v is not None}, prefix='data')
        stats = prefix_name(stats, 'theta')
        stats.update(data)
        stats['theta/popart/mean'] = [rms.mean for rms in self.popart]
        stats['theta/popart/std'] = [rms.std for rms in self.popart]
        with Timer('stats_subsampling'):
            stats = sample_stats(stats, max_record_size=100)

        return stats

    def lookahead_train(self, data: AttrDict):
        theta = self.model.lookahead_params.copy()
        theta.policies, is_lookahead = pop_lookahead(theta.policies)
        assert all([lka == True for lka in is_lookahead]), is_lookahead
        opt_state = self.lookahead_opt_state

        if self.config.update_scheme == 'step':
            theta, opt_state = self.stepwise_sequential_opt(
                theta, opt_state, data, self.config.n_epochs, 
                self.config.n_mbs, self.indices, 
                self.jit_lka_train, return_stats=False
            )
        else:
            theta, opt_state = self.sequential_opt(
                theta, opt_state, data, self.config.n_epochs, 
                self.config.n_mbs, self.indices, 
                self.jit_lka_train, return_stats=False
            )

        for p in theta.policies:
            p[LOOKAHEAD] = True
        self.model.lookahead_params = theta
        self.lookahead_opt_state = opt_state

    def sequential_opt(self, theta, opt_state, data, 
            n_epochs, n_mbs, indices, train_fn, return_stats=True):
        teammate_log_ratio = jnp.zeros_like(data.mu_logprob[:, :, :1])

        v_target = [None for _ in self.aid2uids]
        stats_list = []
        for aid in np.random.permutation(self.n_agents):
            uids = self.aid2uids[aid]
            agent_theta = AttrDict(
                policy=theta.policies[aid], value=theta.vs[aid])
            if self.config.get('theta_opt'):
                agent_opt_state = opt_state[aid]
            else:
                agent_opt_state = AttrDict(
                    policy=opt_state.policies[aid], value=opt_state.vs[aid])
            agent_data = data.slice(indices=uids, axis=2)
            for _ in range(n_epochs):
                vts = []
                np.random.shuffle(indices)
                for idx in np.split(indices, n_mbs):
                    d = agent_data.slice(idx)
                    if self.config.popart:
                        d.popart_mean = self.popart[aid].mean
                        d.popart_std = self.popart[aid].std
                    tlr = teammate_log_ratio[idx]
                    agent_theta, agent_opt_state, stats = \
                        train_fn(
                            agent_theta, 
                            opt_state=agent_opt_state, 
                            data=d, 
                            teammate_log_ratio=tlr, 
                            aid=aid, 
                            compute_teammate_log_ratio=False
                        )
                    vts.append(stats.pop('v_target'))

            teammate_log_ratio = self.compute_teammate_log_ratio(
                agent_theta.policy, self.rng, teammate_log_ratio, agent_data
            )
            
            v_target[aid] = np.concatenate(vts)
            stats.teammate_log_ratio = teammate_log_ratio
            stats_list.append(stats)
            theta.policies[aid] = agent_theta.policy
            theta.vs[aid] = agent_theta.value
            if self.config.get('theta_opt'):
                opt_state[aid] = agent_opt_state
            else:
                opt_state.policies[aid] = agent_opt_state.policy
                opt_state.vs[aid] = agent_opt_state.value
        
        if return_stats:
            stats = batch_dicts(stats_list, np.stack)
            stats.v_target = np.concatenate(v_target, 2)
            assert stats.v_target.shape == data.reward.shape, (stats.v_target.shape, data.reward.shape)
            return theta, opt_state, stats
        return theta, opt_state

    def stepwise_sequential_opt(self, theta, opt_state, data, 
            n_epochs, n_mbs, indices, train_fn, return_stats=True):
        for _ in range(n_epochs):
            np.random.shuffle(indices)
            v_target = []
            stats_list = []
            for idx in np.split(indices, n_mbs):
                vts = [None for _ in self.aid2uids]
                data_slice = data.slice(idx)
                teammate_log_ratio = jnp.zeros_like(data_slice.mu_logprob[:, :, :1])

                for aid in np.random.permutation(self.n_agents):
                    uids = self.aid2uids[aid]
                    agent_theta = AttrDict(
                        policy=theta.policies[aid], value=theta.vs[aid])
                    if self.config.get('theta_opt'):
                        agent_opt_state = opt_state[aid]
                    else:
                        agent_opt_state = AttrDict(
                            policy=opt_state.policies[aid], value=opt_state.vs[aid])
                    agent_data = data_slice.slice(indices=uids, axis=2)
                    if self.config.popart:
                        agent_data.popart_mean = self.popart[aid].mean
                        agent_data.popart_std = self.popart[aid].std
                    agent_theta, agent_opt_state, stats = \
                        train_fn(
                            agent_theta, 
                            opt_state=agent_opt_state, 
                            data=agent_data, 
                            teammate_log_ratio=teammate_log_ratio, 
                            aid=aid,
                            compute_teammate_log_ratio=True
                        )
                    teammate_log_ratio = stats.teammate_log_ratio

                    theta.policies[aid] = agent_theta.policy
                    theta.vs[aid] = agent_theta.value
                    if self.config.get('theta_opt'):
                        opt_state[aid] = agent_opt_state
                    else:
                        opt_state.policies[aid] = agent_opt_state.policy
                        opt_state.vs[aid] = agent_opt_state.value
                    
                    vts[aid] = stats.pop('v_target')
                v_target.append(vts)
                stats_list.append(stats)

        if return_stats:
            stats = batch_dicts(stats_list, np.stack)
            v_target = [np.concatenate(v, 2) for v in v_target]
            stats.v_target = np.concatenate(v_target)
            assert stats.v_target.shape == data.reward.shape, (stats.v_target.shape, data.reward.shape)
            return theta, opt_state, stats
        return theta, opt_state

    def sync_lookahead_params(self):
        self.model.sync_lookahead_params()
        self.lookahead_opt_state = self.params.theta

    def theta_train(
        self, 
        theta, 
        rng, 
        opt_state, 
        data, 
        teammate_log_ratio, 
        aid, 
        compute_teammate_log_ratio=True
    ):
        do_logging('train is traced', backtrack=4)            
        rngs = jax.random.split(rng, 3)
        if self.config.get('theta_opt'):
            theta, opt_state, stats = optimizer.optimize(
                self.loss.loss, 
                theta, 
                opt_state, 
                kwargs={
                    'rng': rngs[0], 
                    'data': data, 
                    'teammate_log_ratio': teammate_log_ratio,
                }, 
                opt=self.opts.theta[aid], 
                name='train/theta'
            )
        else:
            theta.value, opt_state.value, stats = optimizer.optimize(
                self.loss.value_loss, 
                theta.value, 
                opt_state.value, 
                kwargs={
                    'rng': rngs[0], 
                    'policy_theta': theta.policy, 
                    'data': data,
                }, 
                opt=self.opts.vs[aid], 
                name='train/value'
            )
            theta.policy, opt_state.policy, stats = optimizer.optimize(
                self.loss.policy_loss, 
                theta.policy, 
                opt_state.policy, 
                kwargs={
                    'rng': rngs[1], 
                    'data': data, 
                    'stats': stats,
                    'teammate_log_ratio': teammate_log_ratio,
                }, 
                opt=self.opts.policies[aid], 
                name='train/policy'
            )

        if compute_teammate_log_ratio:
            stats.teammate_log_ratio = self.compute_teammate_log_ratio(
                theta.policy, rngs[2], teammate_log_ratio, data)

        return theta, opt_state, stats

    def compute_teammate_log_ratio(
            self, policy_params, rng, teammate_log_ratio, data):
        pi_logprob = self.model.action_logprob(policy_params, rng, data)
        log_ratio = pi_logprob - data.mu_logprob
        if log_ratio.shape[2] > 1:
            log_ratio = jnp.sum(log_ratio, axis=2, keepdims=True)
        teammate_log_ratio += log_ratio
    
        return teammate_log_ratio
    
    def save_optimizer(self):
        super().save_optimizer()
        self.save_popart()
    
    def restore_optimizer(self):
        super().restore_optimizer()
        self.restore_popart()

    def save(self):
        super().save()
        self.save_popart()
    
    def restore(self):
        super().restore()
        self.restore_popart()

    def get_popart_dir(self):
        path = '/'.join([self.config.root_dir, self.config.model_name])
        return path

    def save_popart(self):
        filedir = self.get_popart_dir()
        save(self.popart, filedir=filedir, filename='popart')

    def restore_popart(self):
        filedir = self.get_popart_dir()
        self.popart = restore(
            filedir=filedir, filename='popart', 
            default=[RunningMeanStd((0, 1, 2)) for _ in self.model.aid2uids])

    # def haiku_tabulate(self, data=None):
    #     rng = jax.random.PRNGKey(0)
    #     if data is None:
    #         data = construct_fake_data(self.env_stats, 0)
    #     theta = self.model.theta.copy()
    #     is_lookahead = theta.pop('lookahead')
    #     print(hk.experimental.tabulate(self.theta_train)(
    #         theta, rng, self.params.theta, data
    #     ))
    #     breakpoint()


create_trainer = partial(create_trainer,
    name='happo', trainer_cls=Trainer
)


if __name__ == '__main__':
    import haiku as hk
    from tools.yaml_op import load_config
    from env.func import create_env
    from .model import create_model
    from .loss import create_loss
    from core.log import pwc
    config = load_config('algo/ppo/configs/magw_a2c')
    config = load_config('distributed/sync/configs/smac')
    
    env = create_env(config.env)
    model = create_model(config.model, env.stats())
    loss = create_loss(config.loss, model)
    trainer = create_trainer(config.trainer, env.stats(), loss)
    data = construct_fake_data(env.stats(), 0)
    rng = jax.random.PRNGKey(0)
    pwc(hk.experimental.tabulate(trainer.jit_train)(
        model.theta, rng, trainer.params.theta, data), color='yellow')
    # data = construct_fake_data(env.stats(), 0, True)
    # pwc(hk.experimental.tabulate(trainer.raw_meta_train)(
    #     model.eta, model.theta, trainer.params, data), color='yellow')
