import tensorflow as tf
from tensorflow_probability import distributions as tfd

from core.elements.loss import Loss
from utility.rl_loss import huber_loss, ppo_loss, clipped_value_loss
from utility.tf_utils import assert_rank, assert_rank_and_shape_compatibility, explained_variance, reduce_mean


class PPOLossImpl(Loss):
    def _compute_value_loss(self, value, traj_ret, old_value, mask=None):
        value_loss_type = getattr(self, '_value_loss', 'mse')
        v_clip_frac = 0
        if value_loss_type == 'huber':
            value_loss = reduce_mean(
                huber_loss(value, traj_ret, threshold=self._huber_threshold), mask)
        elif value_loss_type == 'mse':
            value_loss = .5 * reduce_mean((value - traj_ret)**2, mask)
        elif value_loss_type == 'clip':
            value_loss, v_clip_frac = clipped_value_loss(
                value, traj_ret, old_value, self._clip_range, 
                mask=mask)
        elif value_loss_type == 'clip_huber':
            value_loss, v_clip_frac = clipped_value_loss(
                value, traj_ret, old_value, self._clip_range, 
                mask=mask, threshold=self._huber_threshold)
        else:
            raise ValueError(f'Unknown value loss type: {value_loss_type}')
        
        return value_loss, v_clip_frac


class PPOLoss(PPOLossImpl):
    def loss(self, 
            numbers, 
            jokers, 
            left_cards, 
            is_last_teammate_move,
            is_first_move,
            last_valid_action_type,
            rank,
            bombs_dealt,
            last_action_numbers, 
            last_action_jokers, 
            last_action_types,
            last_action_rel_pids,
            last_action_filters, 
            last_action_first_move, 
            action_type_mask, 
            card_rank_mask, 
            mask,
            others_numbers, 
            others_jokers, 
            others_h,
            action_type,
            card_rank,
            state,  
            value,
            traj_ret, 
            advantage, 
            action_type_logpi,
            card_rank_logpi):
        old_value = value
        terms = {}
        with tf.GradientTape() as tape:
            x, _ = self.model.encode(
                numbers=numbers, 
                jokers=jokers, 
                left_cards=left_cards,
                bombs_dealt=bombs_dealt,
                last_action_numbers=last_action_numbers,
                last_action_jokers=last_action_jokers,
                last_action_types=last_action_types,
                last_action_rel_pids=last_action_rel_pids,
                last_action_filters=last_action_filters,
                last_action_first_move=last_action_first_move,
                state=state,
                mask=mask)
            _, card_rank_mask, action_type_dist, card_rank_dist = \
                self.model.compute_policy_stream(
                    x=x, 
                    is_last_teammate_move=is_last_teammate_move,
                    last_valid_action_type=last_valid_action_type, 
                    rank=rank,
                    action_type_mask=action_type_mask,
                    card_rank_mask=card_rank_mask,
                    action_type=action_type,
                    evaluation=False)

            new_action_type_logpi = action_type_dist.log_prob(action_type)
            new_card_rank_logpi = card_rank_dist.log_prob(card_rank)
            action_type_log_ratio = new_action_type_logpi - action_type_logpi
            card_rank_log_ratio = new_card_rank_logpi - card_rank_logpi

            action_type_entropy = action_type_dist.entropy()
            card_rank_entropy = card_rank_dist.entropy()
            
            action_type_loss, action_type_entropy, action_type_kl, action_type_clip_frac = ppo_loss(
                action_type_log_ratio, advantage, self._clip_range, action_type_entropy)
            card_rank_loss, card_rank_entropy, card_rank_kl, card_rank_clip_frac = ppo_loss(
                card_rank_log_ratio, advantage, self._clip_range, card_rank_entropy)
            policy_loss = self._action_type_coef * action_type_loss \
                + self._card_rank_coef * card_rank_loss
            entropy = self._action_type_coef * action_type_entropy \
                + self._card_rank_coef * card_rank_entropy
            actor_loss = (policy_loss - self._entropy_coef * entropy)
            
            # value loss
            value = self.model.compute_value_stream(
                x,
                others_numbers=others_numbers,
                others_jokers=others_jokers,
                others_h=others_h
            )

            value_loss, v_clip_frac = self._compute_value_loss(
                value, traj_ret, old_value)
            value_loss = self._value_coef * value_loss
            loss = actor_loss + value_loss

        terms.update(dict(
            card_rank_mask=card_rank_mask,
            value=value,
            action_type_ratio=tf.exp(action_type_log_ratio), 
            card_rank_ratio=tf.exp(card_rank_log_ratio), 
            action_type_entropy=action_type_entropy, 
            card_rank_entropy=card_rank_entropy, 
            entropy=entropy, 
            action_type_kl=action_type_kl, 
            card_rank_kl=card_rank_kl, 
            action_type_clip_frac=action_type_clip_frac,
            card_rank__clip_frac=card_rank_clip_frac,
            action_type_loss=action_type_loss,
            card_rank_loss=card_rank_loss,
            policy_loss=policy_loss,
            actor_loss=actor_loss,
            v_clip_frac=v_clip_frac,
            v_loss=value_loss,
            explained_variance=explained_variance(traj_ret, value),
        ))

        return tape, loss, terms


class PPGLoss(PPOLossImpl):
    def aux_loss(self,
                numbers, 
                jokers, 
                left_cards, 
                is_last_teammate_move,
                is_first_move,
                last_valid_action_type,
                rank,
                bombs_dealt,
                last_action_numbers, 
                last_action_jokers, 
                last_action_types,
                last_action_rel_pids,
                last_action_filters, 
                last_action_first_move, 
                action_type_mask, 
                card_rank_mask, 
                mask, 
                others_numbers, 
                others_jokers, 
                others_h, 
                action_type,
                state, 
                value,
                traj_ret, 
                action_type_logits,
                card_rank_logits):
        old_value = value
        terms = {}
        with tf.GradientTape() as tape:
            x, _ = self.model.encode(
                numbers=numbers, 
                jokers=jokers, 
                left_cards=left_cards,
                bombs_dealt=bombs_dealt,
                last_action_numbers=last_action_numbers,
                last_action_jokers=last_action_jokers,
                last_action_types=last_action_types,
                last_action_rel_pids=last_action_rel_pids,
                last_action_filters=last_action_filters,
                last_action_first_move=last_action_first_move,
                state=state,
                mask=mask)
            _, card_rank_mask, action_type_dist, card_rank_dist = \
                self.model.compute_policy_stream(
                    x=x, 
                    is_last_teammate_move=is_last_teammate_move,
                    last_valid_action_type=last_valid_action_type, 
                    rank=rank,
                    action_type_mask=action_type_mask,
                    card_rank_mask=card_rank_mask,
                    action_type=action_type,
                    evaluation=False)
            old_action_type_dist = tfd.Categorical(action_type_logits)
            old_card_rank_dist = tfd.Categorical(card_rank_logits)
            action_type_kl = tf.reduce_mean(old_action_type_dist.kl_divergence(action_type_dist))
            card_rank_kl = tf.reduce_mean(old_card_rank_dist.kl_divergence(card_rank_dist))
            bc_loss = self._bc_coef * (action_type_kl + card_rank_kl)

            value = self.model.compute_value_stream(
                x, others_numbers, others_jokers, others_h)
            value_loss, v_clip_frac = self._compute_value_loss(
                value, traj_ret, old_value)
            loss = bc_loss + value_loss
        
        terms = dict(
            value=value,
            action_type_kl=action_type_kl, 
            card_rank_kl=card_rank_kl,
            bc_loss=bc_loss,
            v_loss=value_loss,
            explained_variance=explained_variance(traj_ret, value),
            v_clip_frac=v_clip_frac
        )

        return tape, loss, terms

    def loss(self, 
            numbers, 
            jokers, 
            left_cards, 
            is_last_teammate_move,
            is_first_move,
            last_valid_action_type,
            rank,
            bombs_dealt,
            last_action_numbers, 
            last_action_jokers, 
            last_action_types,
            last_action_rel_pids,
            last_action_filters, 
            last_action_first_move, 
            action_type_mask, 
            card_rank_mask, 
            mask, 
            others_numbers, 
            others_jokers, 
            others_h,
            action_type,
            card_rank,
            state, 
            value,
            traj_ret, 
            advantage, 
            action_type_logpi,
            card_rank_logpi):
        old_value = value
        terms = {}
        with tf.GradientTape() as tape:
            x, _ = self.model.encode(
                numbers=numbers, 
                jokers=jokers, 
                left_cards=left_cards,
                bombs_dealt=bombs_dealt,
                last_action_numbers=last_action_numbers,
                last_action_jokers=last_action_jokers,
                last_action_types=last_action_types,
                last_action_rel_pids=last_action_rel_pids,
                last_action_filters=last_action_filters,
                last_action_first_move=last_action_first_move,
                state=state,
                mask=mask)
            _, card_rank_mask, action_type_dist, card_rank_dist = \
                self.model.compute_policy_stream(
                    x=x, 
                    is_last_teammate_move=is_last_teammate_move,
                    last_valid_action_type=last_valid_action_type, 
                    rank=rank,
                    action_type_mask=action_type_mask,
                    card_rank_mask=card_rank_mask,
                    action_type=action_type,
                    evaluation=False)

            new_action_type_logpi = action_type_dist.log_prob(action_type)
            new_card_rank_logpi = card_rank_dist.log_prob(card_rank)
            action_type_log_ratio = new_action_type_logpi - action_type_logpi
            card_rank_log_ratio = new_card_rank_logpi - card_rank_logpi

            action_type_entropy = action_type_dist.entropy()
            card_rank_entropy = card_rank_dist.entropy()
            
            action_type_loss, action_type_entropy, action_type_kl, action_type_clip_frac = ppo_loss(
                action_type_log_ratio, advantage, self._clip_range, action_type_entropy)
            card_rank_loss, card_rank_entropy, card_rank_kl, card_rank_clip_frac = ppo_loss(
                card_rank_log_ratio, advantage, self._clip_range, card_rank_entropy)
            policy_loss = self._action_type_coef * action_type_loss \
                + self._card_rank_coef * card_rank_loss
            entropy = self._action_type_coef * action_type_entropy \
                + self._card_rank_coef * card_rank_entropy
            actor_loss = (policy_loss - self._entropy_coef * entropy)
            
            # value loss
            x = tf.stop_gradient(x)
            value = self.model.compute_value_stream(
                x,
                others_numbers=others_numbers,
                others_jokers=others_jokers,
                others_h=others_h,
            )

            value_loss, v_clip_frac = self._compute_value_loss(
                value, traj_ret, old_value)
            value_loss = self._value_coef * value_loss
            loss = actor_loss + value_loss

        terms.update(dict(
            card_rank_mask=card_rank_mask,
            value=value,
            action_type_ratio=tf.exp(action_type_log_ratio), 
            card_rank_ratio=tf.exp(card_rank_log_ratio), 
            action_type_entropy=action_type_entropy, 
            card_rank_entropy=card_rank_entropy, 
            entropy=entropy, 
            action_type_kl=action_type_kl, 
            card_rank_kl=card_rank_kl, 
            action_type_clip_frac=action_type_clip_frac,
            card_rank_clip_frac=card_rank_clip_frac,
            action_type_loss=action_type_loss,
            card_rank_loss=card_rank_loss,
            policy_loss=policy_loss,
            actor_loss=actor_loss,
            v_clip_frac=v_clip_frac,
            v_loss=value_loss,
            explained_variance=explained_variance(traj_ret, value),
        ))

        return tape, loss, terms


def create_loss(config, model, name='ppo'):
    if config['training'] == 'ppg' or config['training'] == 'pbt':
        return PPGLoss(config=config, model=model, name=name)
    if config['training'] == 'ppo':
        return PPOLoss(config=config, model=model, name=name)
    else:
        raise ValueError(config['training'])


if __name__ == '__main__':
    import os
    from tensorflow.keras import layers
    from algo.gd.elements.model import create_model
    from env.func import create_env
    from utility.yaml_op import load_config
    config = load_config('algo/gd/configs/guandan.yaml')
    env = create_env(config['env'])
    env_stats = env.stats()
    model = create_model(config['model'], env_stats)
    loss = create_loss(config['loss'], model)
    b = 2
    s = 3
    shapes = {
        **env_stats['obs_shape'], 
        **env_stats['action_shape'],
        'state': model.get_initial_state(),
    }
    x = {k: layers.Input(v) for k, v in inp.items()}
    y = loss.loss(**x)
    model = tf.keras.Model(x, y)
    model.summary(200)
