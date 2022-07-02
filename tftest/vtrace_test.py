"""Tests for v-retrace."""

# Dependency imports
import numpy as np
import tensorflow as tf

from utility.rl_loss import v_trace


def _shaped_arange(*shape):
    """Runs np.arange, converts to float and reshapes."""
    return np.arange(np.prod(shape), dtype=np.float32).reshape(*shape)

def _softmax(logits):
    """Applies softmax non-linearity on inputs."""
    return np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

def _ground_truth_calculation(reward, value, next_value, pi, mu, discount, gamma=1, lambda_=1,
        c_clip=1, rho_clip=1, rho_clip_pg=1, axis=0):
    log_rhos = tf.convert_to_tensor(tf.math.log(pi/mu), dtype=tf.float32)
    discounts = tf.convert_to_tensor(discount, dtype=tf.float32)
    rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
    values = tf.convert_to_tensor(value, dtype=tf.float32)
    bootstrap_value = tf.convert_to_tensor(next_value[-1], dtype=tf.float32)
    if rho_clip is not None:
        clip_rho_threshold = tf.convert_to_tensor(rho_clip, dtype=tf.float32)
    if rho_clip_pg is not None:
        clip_pg_rho_threshold = tf.convert_to_tensor(rho_clip_pg, dtype=tf.float32)

    # Make sure tensor ranks are consistent.
    rho_rank = log_rhos.shape.ndims  # Usually 2.
    values.shape.assert_has_rank(rho_rank)
    bootstrap_value.shape.assert_has_rank(rho_rank - 1)
    discounts.shape.assert_has_rank(rho_rank)
    rewards.shape.assert_has_rank(rho_rank)
    if clip_rho_threshold is not None:
        clip_rho_threshold.shape.assert_has_rank(0)
    if clip_pg_rho_threshold is not None:
        clip_pg_rho_threshold.shape.assert_has_rank(0)

    rhos = tf.exp(log_rhos)
    if clip_rho_threshold is not None:
        clipped_rhos = tf.minimum(clip_rho_threshold, rhos, name='clipped_rhos')
    else:
        clipped_rhos = rhos

    cs = lambda_ * tf.minimum(c_clip, rhos, name='cs')
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = tf.concat(
        [values[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
    discounts = discounts * gamma
    deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

    # Note that all sequences are reversed, computation starts from the back.
    sequences = (
        tf.reverse(discounts, axis=[0]),
        tf.reverse(cs, axis=[0]),
        tf.reverse(deltas, axis=[0]),
    )
    # V-trace vs are calculated through a scan from the back to the beginning
    # of the given trajectory.
    def scanfunc(acc, sequence_item):
        discount_t, c_t, delta_t = sequence_item
        return delta_t + discount_t * c_t * acc

    initial_values = tf.zeros_like(bootstrap_value)
    vs_minus_v_xs = tf.scan(
        fn=scanfunc,
        elems=sequences,
        initializer=initial_values,
        parallel_iterations=1,
        back_prop=False,
        name='scan')
    # Reverse the results back to original order.
    vs_minus_v_xs = tf.reverse(vs_minus_v_xs, [0], name='vs_minus_v_xs')

    # Add V(x_s) to get v_s.
    vs = tf.add(vs_minus_v_xs, values, name='vs')

    # Advantage for policy gradient.
    vs_t_plus_1 = tf.concat([
        vs[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
    if clip_pg_rho_threshold is not None:
        clipped_pg_rhos = tf.minimum(clip_pg_rho_threshold, rhos,
                                   name='clipped_pg_rhos')
    else:
        clipped_pg_rhos = rhos
    pg_advantages = (
        clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values))

    return vs, pg_advantages


class VRetraceTest(tf.test.TestCase):
    """Tests for `Retrace` ops."""

    def testVTrace(self):
        """Tests V-trace against ground truth data calculated in python."""
        for batch_size in [1, 5]:
            seq_len = 5

            values = _shaped_arange(seq_len+1, batch_size) / batch_size
            value = values[:-1]
            next_value = values[1:]
            values = {
                # Note that this is only for testing purposes using well-formed inputs.
                # In practice we'd be more careful about taking log() of arbitrary
                # quantities.
                'reward': _shaped_arange(seq_len, batch_size),
                'value': value,
                'next_value': next_value,
                'pi': np.random.uniform(0, 1, size=(seq_len, batch_size)),
                'mu': np.random.uniform(0, 1, size=(seq_len, batch_size)),
                # T, B where B_i: [0.9 / (i+1)] * T
                'discount': np.array([[0.9 / (b + 1)
                            for b in range(batch_size)]
                            for _ in range(seq_len)]),
                'c_clip': 0.7,
                'rho_clip': 1.2,
                'rho_clip_pg': 2.1,
            }

            output_v = v_trace(**tf.nest.map_structure(
                lambda x: tf.convert_to_tensor(x, tf.float32), values))

            ground_truth_v = _ground_truth_calculation(**tf.nest.map_structure(
                lambda x: tf.convert_to_tensor(x, tf.float32), values))
            for a, b in zip(ground_truth_v, output_v):
                self.assertAllClose(a, b)

    def testVTraceAxis1(self):
        """Tests V-trace against ground truth data calculated in python."""
        for batch_size in [1, 5]:
            seq_len = 5

            values = _shaped_arange(seq_len+1, batch_size) / batch_size
            value = values[:-1]
            next_value = values[1:]
            values = {
                # Note that this is only for testing purposes using well-formed inputs.
                # In practice we'd be more careful about taking log() of arbitrary
                # quantities.
                'reward': _shaped_arange(seq_len, batch_size),
                'value': value,
                'next_value': next_value,
                'pi': np.random.uniform(0, 1, size=(seq_len, batch_size)),
                'mu': np.random.uniform(0, 1, size=(seq_len, batch_size)),
                # T, B where B_i: [0.9 / (i+1)] * T
                'discount': np.array([[0.9 / (b + 1)
                            for b in range(batch_size)]
                            for _ in range(seq_len)]),
                'gamma': .99, 
                'lambda_': .95, 
                'c_clip': 3.7,
                'rho_clip': 2.2,
                'rho_clip_pg': 4.1,
            }

            output_v = v_trace(**tf.nest.map_structure(
                lambda x: tf.convert_to_tensor(np.swapaxes(x, 0, 1) if isinstance(x, np.ndarray) else x, 
                tf.float32), values), axis=1)
            output_v = tf.nest.map_structure(lambda x: tf.transpose(x, [1, 0]), output_v)
            ground_truth_v = _ground_truth_calculation(**tf.nest.map_structure(
                lambda x: tf.convert_to_tensor(x, tf.float32), values))
            for a, b in zip(ground_truth_v, output_v):
                self.assertAllClose(a, b)

#   @parameterized.named_parameters(('Batch1', 1), ('Batch2', 2))
#   def testVTraceFromLogits(self, batch_size):
#     """Tests V-trace calculated from logits."""
#     seq_len = 5
#     num_actions = 3
#     clip_rho_threshold = None  # No clipping.
#     clip_pg_rho_threshold = None  # No clipping.

#     # Intentionally leaving shapes unspecified to test if V-trace can
#     # deal with that.
#     placeholders = {
#         # T, B, NUM_ACTIONS
#         'behaviour_policy_logits':
#             tf.placeholder(dtype=tf.float32, shape=[None, None, None]),
#         # T, B, NUM_ACTIONS
#         'target_policy_logits':
#             tf.placeholder(dtype=tf.float32, shape=[None, None, None]),
#         'actions':
#             tf.placeholder(dtype=tf.int32, shape=[None, None]),
#         'discounts':
#             tf.placeholder(dtype=tf.float32, shape=[None, None]),
#         'reward':
#             tf.placeholder(dtype=tf.float32, shape=[None, None]),
#         'values':
#             tf.placeholder(dtype=tf.float32, shape=[None, None]),
#         'bootstrap_value':
#             tf.placeholder(dtype=tf.float32, shape=[None]),
#     }

#     from_logits_output = vtrace_ops.vtrace_from_logits(
#         clip_rho_threshold=clip_rho_threshold,
#         clip_pg_rho_threshold=clip_pg_rho_threshold,
#         **placeholders)

#     target_log_probs = vtrace_ops.log_probs_from_logits_and_actions(
#         placeholders['target_policy_logits'], placeholders['actions'])
#     behaviour_log_probs = vtrace_ops.log_probs_from_logits_and_actions(
#         placeholders['behaviour_policy_logits'], placeholders['actions'])
#     log_rhos = target_log_probs - behaviour_log_probs
#     ground_truth = (log_rhos, behaviour_log_probs, target_log_probs)

#     values = {
#         'behaviour_policy_logits':
#             _shaped_arange(seq_len, batch_size, num_actions),
#         'target_policy_logits':
#             _shaped_arange(seq_len, batch_size, num_actions),
#         'actions':
#             np.random.randint(0, num_actions - 1, size=(seq_len, batch_size)),
#         'discounts':
#             np.array(  # T, B where B_i: [0.9 / (i+1)] * T
#                 [[0.9 / (b + 1)
#                   for b in range(batch_size)]
#                  for _ in range(seq_len)]),
#         'reward':
#             _shaped_arange(seq_len, batch_size),
#         'values':
#             _shaped_arange(seq_len, batch_size) / batch_size,
#         'bootstrap_value':
#             _shaped_arange(batch_size) + 1.0,  # B
#     }

#     feed_dict = {placeholders[k]: v for k, v in values.items()}
#     with self.test_session() as session:
#       from_logits_output_v = session.run(
#           from_logits_output, feed_dict=feed_dict)
#       (ground_truth_log_rhos, ground_truth_behaviour_action_log_probs,
#        ground_truth_target_action_log_probs) = session.run(
#            ground_truth, feed_dict=feed_dict)

#     # Calculate V-trace using the ground truth logits.
#     from_iw = vtrace_ops.vtrace_from_importance_weights(
#         log_rhos=ground_truth_log_rhos,
#         discounts=values['discounts'],
#         reward=values['reward'],
#         values=values['values'],
#         bootstrap_value=values['bootstrap_value'],
#         clip_rho_threshold=clip_rho_threshold,
#         clip_pg_rho_threshold=clip_pg_rho_threshold)

#     with self.test_session() as session:
#       from_iw_v = session.run(from_iw)

#     self.assertAllClose(from_iw_v.vs, from_logits_output_v.vs)
#     self.assertAllClose(from_iw_v.pg_advantages,
#                         from_logits_output_v.pg_advantages)
#     self.assertAllClose(ground_truth_behaviour_action_log_probs,
#                         from_logits_output_v.behaviour_action_log_probs)
#     self.assertAllClose(ground_truth_target_action_log_probs,
#                         from_logits_output_v.target_action_log_probs)
#     self.assertAllClose(ground_truth_log_rhos, from_logits_output_v.log_rhos)

#   def testHigherRankInputsForIW(self):
#     """Checks support for additional dimensions in inputs."""
#     placeholders = {
#         'log_rhos': tf.placeholder(dtype=tf.float32, shape=[None, None, 1]),
#         'discounts': tf.placeholder(dtype=tf.float32, shape=[None, None, 1]),
#         'reward': tf.placeholder(dtype=tf.float32, shape=[None, None, 42]),
#         'values': tf.placeholder(dtype=tf.float32, shape=[None, None, 42]),
#         'bootstrap_value': tf.placeholder(dtype=tf.float32, shape=[None, 42])
#     }
#     output = vtrace_ops.vtrace_from_importance_weights(**placeholders)
#     self.assertEqual(output.vs.shape.as_list()[-1], 42)

#   def testInconsistentRankInputsForIW(self):
#     """Test one of many possible errors in shape of inputs."""
#     placeholders = {
#         'log_rhos': tf.placeholder(dtype=tf.float32, shape=[None, None, 1]),
#         'discounts': tf.placeholder(dtype=tf.float32, shape=[None, None, 1]),
#         'reward': tf.placeholder(dtype=tf.float32, shape=[None, None, 42]),
#         'values': tf.placeholder(dtype=tf.float32, shape=[None, None, 42]),
#         # Should be [None, 42].
#         'bootstrap_value': tf.placeholder(dtype=tf.float32, shape=[None])
#     }
#     with self.assertRaisesRegexp(ValueError, 'must have rank 2'):
#       vtrace_ops.vtrace_from_importance_weights(**placeholders)

if __name__ == '__main__':
  tf.test.main()
