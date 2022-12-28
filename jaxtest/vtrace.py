import functools
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import numpy as np
import rlax

from core.typing import dict2AttrDict
from jax_tools.jax_loss import compute_target_advantage

class VTraceTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()

        behavior_policy_logits = np.array(
            [[[8.9, 0.7], [5.0, 1.0], [0.6, 0.1], [-0.9, -0.1]],
            [[0.3, -5.0], [1.0, -8.0], [0.3, 1.7], [4.7, 3.3]]],
            dtype=np.float32)
        target_policy_logits = np.array(
            [[[0.4, 0.5], [9.2, 8.8], [0.7, 4.4], [7.9, 1.4]],
            [[1.0, 0.9], [1.0, -1.0], [-4.3, 8.7], [0.8, 0.3]]],
            dtype=np.float32)
        actions = np.array([[0, 1, 0, 0], [1, 0, 0, 1]], dtype=np.int32)
        self._rho_tm1 = rlax.categorical_importance_sampling_ratios(
            target_policy_logits, behavior_policy_logits, actions)
        self._rewards = np.array(
            [[-1.3, -1.3, 2.3, 42.0],
            [1.3, 5.3, -3.3, -5.0]],
            dtype=np.float32)
        self._discounts = np.array(
            [[0., 0.89, 0.85, 0.99],
            [0.88, 1., 0.83, 0.95]],
            dtype=np.float32)
        self._values = np.array(
            [[2.1, 1.1, -3.1, 0.0],
            [3.1, 0.1, -1.1, 7.4]],
            dtype=np.float32)
        self._bootstrap_value = np.array([8.4, -1.2], dtype=np.float32)
        self._inputs = [
            self._rewards, self._discounts, self._rho_tm1,
            self._values, self._bootstrap_value]

        self._clip_rho_threshold = 1.0
        self._clip_pg_rho_threshold = 5.0
        self._lambda = 1.0

        self._expected_td = np.array(
            [[-1.6155143, -3.4973226, 1.8670533, 5.0316002e1],
            [1.4662437, 3.6116405, -8.3327293e-5, -1.3540000e1]],
            dtype=np.float32)
        self._expected_pg = np.array(
            [[-1.6155143, -3.4973226, 1.8670534, 5.0316002e1],
            [1.4662433, 3.6116405, -8.3369283e-05, -1.3540000e+1]],
            dtype=np.float32)

    # @chex.all_variants()
    def test_vtrace_td_error_and_advantage(self):
        """Tests for a full batch."""
        config = dict(
            target_type='vtrace', 
            c_clip=1, 
            rho_clip=self._clip_rho_threshold, 
        )
        config = dict2AttrDict(config)
        my_vtrace = compute_target_advantage
        # my_vtrace = v_trace_from_ratio
        vs, adv = my_vtrace(
            config=config, 
            reward=self._rewards, 
            discount=self._discounts, 
            value=self._values, 
            next_value=np.concatenate([self._values[:, 1:], self._bootstrap_value[:, None]], axis=1), 
            ratio=self._rho_tm1, 
            gamma=1, 
            lam=self._lambda, 
            axis=1
        )
        # Test output.
        np.testing.assert_allclose(
            self._expected_td, vs - self._values, rtol=1e-3)
        np.testing.assert_allclose(
            self._expected_pg, adv, rtol=1e-3)

        # my_vtrace = v_trace_from_ratio
        config.rho_clip_pg = self._clip_pg_rho_threshold
        vs, adv = my_vtrace(
            config=config, 
            reward=self._rewards, 
            discount=self._discounts, 
            value=self._values, 
            next_value=np.concatenate([self._values[:, 1:], self._bootstrap_value[:, None]], axis=1), 
            ratio=self._rho_tm1, 
            gamma=1, 
            lam=self._lambda
        )
        vtrace_td_error_and_advantage = jax.vmap(functools.partial(
            rlax.vtrace_td_error_and_advantage,
            clip_rho_threshold=self._clip_rho_threshold, 
            clip_pg_rho_threshold=self._clip_pg_rho_threshold, 
            lambda_=self._lambda))
        # vtrace_td_error_and_advantage = jax.vmap(functools.partial(
        #     rlax.vtrace_td_error_and_advantage,
        #     clip_rho_threshold=self._clip_rho_threshold, lambda_=self._lambda))
        r_t, discount_t, rho_tm1, v_tm1, bootstrap_value = self._inputs
        v_t = np.concatenate([v_tm1[:, 1:], bootstrap_value[:, None]], axis=1)
        vtrace_output = vtrace_td_error_and_advantage(
            v_tm1, v_t, r_t, discount_t, rho_tm1)
        np.testing.assert_allclose(
            vtrace_output.errors, vs - self._values, rtol=1e-3)
        np.testing.assert_allclose(
            vtrace_output.pg_advantage, adv, rtol=1e-3)

if __name__ == '__main__':
    jax.config.update('jax_numpy_rank_promotion', 'raise')
    absltest.main()