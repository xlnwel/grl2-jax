import tensorflow as tf

from algo.gpo.elements.loss import ValueLossImpl, GPOLossImpl


class GPOLoss(ValueLossImpl, GPOLossImpl):
    def loss(
        self, 
        obs, 
        action, 
        value, 
        traj_ret, 
        advantage, 
        logprob, 
        target_prob, 
        tr_prob, 
        target_prob_prime, 
        tr_prob_prime, 
        pi=None, 
        target_pi=None, 
        pi_mean=None, 
        pi_std=None, 
        state=None, 
        action_mask=None, 
        life_mask=None, 
        mask=None
    ):
        old_value = value
        sample_mask = life_mask
        n = None if sample_mask is None else tf.reduce_sum(sample_mask)
        with tf.GradientTape() as tape:
            x, _ = self.model.encode(
                x=obs, 
                state=state, 
                mask=mask
            )

            act_dist = self.policy(x)
            actor_terms, actor_loss = self._ppo_loss(
                tape=tape, 
                act_dist=act_dist, 
                action=action, 
                advantage=advantage, 
                logprob=logprob, 
                tr_prob=tr_prob, 
                target_prob_prime=target_prob_prime,
                tr_prob_prime=tr_prob_prime,
                pi=pi, 
                target_pi=target_pi, 
                pi_mean=pi_mean, 
                pi_std=pi_std, 
                action_mask=action_mask,
                sample_mask=sample_mask,
                n=n
            )

            value = self.value(x)
            value_terms, value_loss = self._value_loss(
                tape=tape, 
                value=value,
                traj_ret=traj_ret, 
                old_value=old_value,
                sample_mask=sample_mask, 
                n=n, 
            )

            loss = actor_loss + value_loss
        
        terms = {**actor_terms, **value_terms}
        if self.config.get('debug', True) and life_mask is not None:
                terms['n_alive_units'] = tf.reduce_sum(
                    life_mask, -1)

        return tape, loss, terms

def create_loss(config, model, name='gpo'):
    return GPOLoss(config=config, model=model, name=name)
