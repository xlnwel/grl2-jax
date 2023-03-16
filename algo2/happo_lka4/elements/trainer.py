from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

from core.elements.trainer import create_trainer
from tools.timer import Timer
from algo.happo.elements.trainer import Trainer as TrainerBase


class Trainer(TrainerBase):
    def fake_lookahead_train(self, data, teammate_log_ratio=None):
        if teammate_log_ratio is None:
            teammate_log_ratio = jnp.zeros_like(data.mu_logprob)

        theta = self.model.lookahead_params.copy()
        is_lookahead = theta.pop('lookahead')
        assert is_lookahead == True, is_lookahead
        opt_state = self.lookahead_opt_state
        for _ in range(self.config.n_lka_epochs):
            np.random.shuffle(self.indices)
            indices = np.split(self.indices, self.config.n_mbs)
            for idx in indices:
                with Timer('lookahead_train'):
                    d = data.slice(idx)
                    t_log_ratio = teammate_log_ratio[idx]
                    if self.config.popart:
                        d.popart_mean = self.popart.mean
                        d.popart_std = self.popart.std
                    theta, opt_state, _ = \
                        self.jit_lka_train(
                            theta, 
                            opt_state=opt_state, 
                            data=d,
                            teammate_log_ratio=t_log_ratio,
                        )

        self.rng, rng = jax.random.split(self.rng) 
        pi_logprob = self.model.jit_action_logprob(
            self.model.lookahead_params, rng, data)
        agent_log_ratio = pi_logprob - data.mu_logprob
        return teammate_log_ratio + agent_log_ratio


create_trainer = partial(create_trainer,
    name='happo', trainer_cls=Trainer
)
