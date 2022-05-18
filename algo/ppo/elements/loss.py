import tensorflow as tf

from algo.gpo.elements.loss import ValueLossImpl, PGLossImpl


class PPOLoss(ValueLossImpl, PGLossImpl):
    def loss(
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
        action_mask=None, 
        life_mask=None, 
        mask=None
    ):
        old_value = value
        loss_mask = life_mask
        n = None if loss_mask is None else tf.reduce_sum(loss_mask)
        with tf.GradientTape() as tape:
            x, _ = self.model.encode(
                x=obs, 
                state=state, 
                mask=mask
            )

            act_dist = self.policy(x)
            actor_terms, actor_loss = self._pg_loss(
                tape=tape, 
                act_dist=act_dist, 
                action=action, 
                advantage=advantage, 
                tr_prob=tr_prob, 
                vt_prob=vt_prob, 
                logprob=logprob, 
                target_pi=target_pi, 
                pi=pi, 
                pi_mean=pi_mean, 
                pi_std=pi_std, 
                action_mask=action_mask,
                mask=loss_mask,
                n=n
            )

            value = self.value(x)
            value_terms, value_loss = self._value_loss(
                tape=tape, 
                value=value,
                traj_ret=traj_ret, 
                old_value=old_value,
                mask=loss_mask, 
                n=n, 
            )

            loss = actor_loss + value_loss
        
        terms = {**actor_terms, **value_terms}
        if self.config.get('debug', True):
            terms.update(dict(
                t_old_diff_prob=target_prob - terms['prob'], 
                tp_old_diff_prob=target_prob_prime - terms['prob'], 
                trp_old_diff_prob=tr_prob_prime - terms['prob'], 
                tp_t_diff_prob=target_prob_prime - target_prob, 
                trp_tr_diff_prob=tr_prob_prime - tr_prob, 
            ))
            if life_mask is not None:
                terms['n_alive_units'] = tf.reduce_sum(
                    life_mask, -1)

        return tape, loss, terms

def create_loss(config, model, name='ppo'):
    return PPOLoss(config=config, model=model, name=name)
