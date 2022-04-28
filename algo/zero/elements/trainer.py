from core.elements.trainer import TrainerEnsemble
from core.decorator import override
from core.tf_config import build
from .utils import get_data_format
from algo.hm.elements.trainer import PPOActorTrainer, PPOValueTrainer


class PPOTrainerEnsemble(TrainerEnsemble):
    @override(TrainerEnsemble)
    def _build_train(self, env_stats):
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = get_data_format(self.config, env_stats, self.model)
        self.train = build(self.train, TensorSpecs)
        return True

    def raw_train(
        self, 
        obs, 
        action, 
        value, 
        value_a, 
        traj_ret, 
        traj_ret_a, 
        advantage, 
        target_prob, 
        tr_prob, 
        logprob, 
        pi=None, 
        pi_mean=None, 
        pi_std=None, 
        global_state=None, 
        reward=None, 
        raw_adv=None, 
        prev_reward=None, 
        prev_action=None, 
        action_mask=None, 
        life_mask=None, 
        actor_state=None, 
        value_state=None, 
        mask=None
    ):
        actor_terms = self.policy.raw_train(
            obs=obs, 
            action=action, 
            advantage=advantage, 
            target_prob=target_prob, 
            tr_prob=tr_prob, 
            logprob=logprob, 
            pi=pi,
            pi_mean=pi_mean, 
            pi_std=pi_std, 
            prev_reward=prev_reward, 
            prev_action=prev_action, 
            state=actor_state, 
            action_mask=action_mask, 
            life_mask=life_mask, 
            mask=mask
        )

        value_terms = self.value.raw_train(
            global_state=global_state, 
            action=action, 
            pi=actor_terms['new_pi'], 
            value=value, 
            value_a=value_a, 
            traj_ret=traj_ret, 
            traj_ret_a=traj_ret_a, 
            prev_reward=prev_reward, 
            prev_action=prev_action, 
            state=value_state, 
            life_mask=life_mask,
            mask=mask
        )

        return {**actor_terms, **value_terms}


def create_trainer(config, env_stats, loss, name='ppo'):
    def constructor(config, env_stats, cls, name):
        return cls(
            config=config, 
            env_stats=env_stats, 
            loss=loss[name], 
            name=name)

    return PPOTrainerEnsemble(
        config=config,
        env_stats=env_stats,
        loss=loss,
        constructor=constructor,
        name=name,
        policy=PPOActorTrainer,
        value=PPOValueTrainer,
    )
