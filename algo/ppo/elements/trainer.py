import functools

from core.elements.trainer import Trainer, create_trainer
from core.decorator import override
from core.tf_config import build
from utility import pkg


class PPOTrainer(Trainer):
    @override(Trainer)
    def _build_train(self, env_stats):
        algo = self.config.algorithm.split('-')[-1]
        get_data_format = pkg.import_module(
            'elements.utils', algo=algo).get_data_format
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = get_data_format(self.config, env_stats, self.loss.model)
        self.train = build(self.train, TensorSpecs)
        return True

    def raw_train(
        self, 
        obs, 
        action, 
        value, 
        traj_ret, 
        advantage, 
        target_prob, 
        tr_prob, 
        vt_prob, 
        target_prob_prime, 
        tr_prob_prime, 
        logprob, 
        pi=None, 
        target_pi=None, 
        pi_mean=None, 
        pi_std=None, 
        state=None, 
        mask=None
    ):
        tape, loss, terms = self.loss.loss(
            obs=obs, 
            action=action, 
            value=value, 
            traj_ret=traj_ret, 
            advantage=advantage, 
            target_prob=target_prob, 
            tr_prob=tr_prob, 
            vt_prob=vt_prob, 
            target_prob_prime=target_prob_prime, 
            tr_prob_prime=tr_prob_prime, 
            logprob=logprob, 
            pi=pi, 
            target_pi=target_pi, 
            pi_mean=pi_mean, 
            pi_std=pi_std, 
            state=state, 
            mask=mask
        )

        terms['norm'], terms['var_norm'] = \
            self.optimizer(tape, loss, return_var_norms=True)

        return terms


create_trainer = functools.partial(create_trainer,
    name='ppo', trainer_cls=PPOTrainer
)
