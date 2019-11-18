from multiprocessing import Process
import numpy as np
import tensorflow as tf

from utility.display import pwc, assert_colorize
from utility.timer import Timer
from utility.tf_utils import build, configure_gpu
from core.base import BaseAgent, agent_config
from algo.ppo.nn import PPOAC


class Agent(BaseAgent):
    @agent_config
    def __init__(self, name, config, env, buffer, models):
        self.gae_discount = self.gamma * self.lam

        self.state_shape = self.env.state_shape
        self.action_shape = self.env.action_shape
        self.action_dim = self.env.action_dim
        self.max_epslen = self.env.max_episode_steps

        self.n_envs = env.n_envs

        # optimizer
        if self.optimizer.lower() == 'adam':
            optimizer = tf.keras.optimizers.Adam
        elif self.optimizer.lower() == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop
        else:
            raise NotImplementedError()
        self.optimizer = optimizer(learning_rate=self.learning_rate,
                                    epsilon=self.epsilon)
        self.ckpt_models['optimizer'] = self.optimizer

        # Explicitly instantiate tf.function to avoid unintended retracing
        
        TensorSpecs = [
            (self.state_shape, self.env.state_dtype, 'state'),
            (self.action_shape, self.env.action_dtype, 'action'),
            ([1], tf.float32, 'traj_ret'),
            ([1], tf.float32, 'value'),
            ([1], tf.float32, 'advantage'),
            ([1], tf.float32, 'old_logpi'),
            ([1], tf.float32, 'mask'),
            ((), tf.float32, 'n'),
        ]
        self.compute_gradients = build(
            self._compute_gradients, 
            TensorSpecs, 
            sequential=True, 
            batch_size=self.n_envs
        )

        # process used for evaluation
        self.eval_process = None

    def train(self):
        period = 10
        profiler_outdir = f'{self.log_root_dir}/{self.model_name}'
        start_epoch = self.global_steps.numpy()
        for epoch in range(start_epoch, self.n_epochs+1):
            self.set_summary_step(epoch)
            with Timer(f'{self.model_name} sampling', period):
                score, epslen, last_value = self._sample_trajectories(epoch)
            with Timer(f'{self.model_name} advantage', period):
                self._compute_advantages(last_value)
            # tf.summary.trace_on(profiler=True)
            with Timer(f'{self.model_name} training', period):
                # TRICK: we only check kl and early terminate the training epoch 
                # when score meets some requirement
                self._train_epoch(epoch, early_terminate=(self.max_kl and score > 250))
            # with Timer('trace'):
            #     tf.summary.trace_export(name='ppo', step=epoch, profiler_outdir=profiler_outdir)
            if epoch % period == 0:
                with Timer(f'{self.model_name} logging'):
                    self.log(epoch, 'Train')
                with Timer(f'{self.model_name} save'):
                    self.save(steps=epoch)
                # self._evaluate(epoch)

    def _sample_trajectories(self, epoch):
        self.buffer.reset()
        self.ac.reset_states()
        env = self.env
        state = env.reset()

        for _ in range(self.max_epslen):
            action, logpi, value = self.ac(tf.convert_to_tensor(state, tf.float32))
            next_state, reward, done, _ = env.step(action.numpy())
            self.buffer.add(state=state, 
                            action=action.numpy(), 
                            reward=reward, 
                            value=value.numpy(), 
                            old_logpi=logpi.numpy(), 
                            nonterminal=1-done,
                            mask=env.get_mask())
            
            state = next_state
            if np.all(done):
                break
            
        _, _, last_value = self.ac(state)

        score, epslen = env.get_score(), env.get_epslen()
        # tf.summary.histogram('epslen', epslen, step=epoch)
        score_mean = np.mean(score)
        epslen_mean = np.mean(epslen)
        self.store(score=score_mean,
                   score_std=np.std(score),
                   epslen=epslen_mean,
                   epslen_std=np.std(epslen))

        return score_mean, epslen_mean, last_value

    def _compute_advantages(self, last_value):
        self.buffer.finish(last_value.numpy(), 
                            self.advantage_type, 
                            self.gamma, 
                            self.gae_discount)

    def _train_epoch(self, epoch, early_terminate):
        for i in range(self.n_updates):
            self.ac.reset_states()
            for j in range(self.n_minibatches):
                data = self.buffer.get_batch()
                data['n'] = n = np.sum(data['mask'])
                value = np.mean(data['value'])
                data = {k: tf.convert_to_tensor(v) for k, v in data.items()}
                with tf.name_scope('train'):
                    loss_info  = self.compute_gradients(**data)
                    entropy, approx_kl, p_clip_frac, v_clip_frac, ppo_loss, value_loss, grads = loss_info
                    if hasattr(self, 'clip_norm'):
                        grads, global_norm = tf.clip_by_global_norm(grads, self.clip_norm)
                    self.optimizer.apply_gradients(zip(grads, self.ac.trainable_variables))

                n_total_trans = np.prod(data['value'].shape)
                n_valid_trans = n or n_total_trans
                self.store(ppo_loss=ppo_loss.numpy(), 
                            value_loss=value_loss.numpy(),
                            entropy=entropy.numpy(), 
                            p_clip_frac=p_clip_frac.numpy(),
                            v_clip_frac=v_clip_frac.numpy(),
                            value=value,
                            global_norm=global_norm.numpy(),
                            n_valid_trans=n_valid_trans,
                            n_total_trans=n_total_trans,
                            valid_trans_frac = n_valid_trans / n_total_trans
                            )
            
            if self.max_kl and early_terminate and approx_kl > self.max_kl:
                pwc(f'Eearly stopping at epoch-{epoch} update-{i+1} due to reaching max kl.',
                    f'Current kl={approx_kl:.3g}', color='blue')
                break
        self.store(approx_kl=approx_kl)

    @tf.function
    def _compute_gradients(self, state, action, traj_ret, value, advantage, old_logpi, mask=None, n=None):
        # self.ac.reset_states()
        with tf.GradientTape() as tape:
            logpi, entropy, v = self.ac.train_step(state, action)
            loss_info = self._loss(logpi, 
                                    old_logpi,
                                    advantage,
                                    v, 
                                    traj_ret, 
                                    value,
                                    self.clip_range,
                                    entropy,
                                    mask=mask,
                                    n=n)
            ppo_loss, entropy, approx_kl, p_clip_frac, value_loss, v_clip_frac, total_loss = loss_info

        with tf.name_scope('gradient'):
            grads = tape.gradient(total_loss, self.ac.trainable_variables)

        return entropy, approx_kl, p_clip_frac, v_clip_frac, ppo_loss, value_loss, grads 

    def _loss(self, logpi, old_logpi, advantages, value, traj_ret, old_value, clip_range, entropy, mask=None, n=None):
        assert (mask is None) == (n is None), f'Both/Neither mask and/nor n should be None, but get \nmask:{mask}\nn:{n}'

        def reduce_mean(x, name, n):
            with tf.name_scope(name):        
                return tf.reduce_mean(x) if n is None else tf.reduce_sum(x) / n

        m = 1. if mask is None else mask
        with tf.name_scope('ppo_loss'):
            ratio = tf.exp(logpi - old_logpi, name='ratio')
            loss1 = -advantages * ratio
            loss2 = -advantages * tf.clip_by_value(ratio, 1. - clip_range, 1. + clip_range)
            
            ppo_loss = reduce_mean(tf.maximum(loss1, loss2) * m, 'ppo_loss', n)
            entropy = tf.reduce_mean(entropy, name='entropy_loss')
            # debug stats: KL between old and current policy and fraction of data being clipped
            approx_kl = .5 * reduce_mean((old_logpi - logpi)**2 * m, 'approx_kl', n)
            p_clip_frac = reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.), clip_range), tf.float32) * m, 
                                    'clip_frac', n)
            policy_loss = (ppo_loss 
                        - self.entropy_coef * entropy # TODO: adaptive entropy regularizer
                        + self.kl_coef * approx_kl)

        with tf.name_scope('value_loss'):
            value_clipped = old_value + tf.clip_by_value(value - old_value, -clip_range, clip_range)
            loss1 = (value - traj_ret)**2
            loss2 = (value_clipped - traj_ret)**2
            
            value_loss = self.value_coef * reduce_mean(tf.maximum(loss1, loss2) * m, 'value_loss', n)
            v_clip_frac = reduce_mean(
                tf.cast(tf.greater(tf.abs(value-old_value), clip_range), tf.float32) * m,
                'clip_frac', n)
        
        with tf.name_scope('total_loss'):
            total_loss = policy_loss + value_loss

        return ppo_loss, entropy, approx_kl, p_clip_frac, value_loss, v_clip_frac, total_loss

    def _evaluate(self, train_epoch):
        """
        Evaluate the learned policy in another process. 
        This function should only be called at the training time.
        """
        if self.eval_process:
            self.eval_process.join()    # join the previous running eval_process
        self.eval_process = Process(target=eval_process, 
            args=(self.env, self.ac.get_weights(), self.model_name, self.logger, train_epoch))
        self.eval_process.start()

def eval_process(env, weights, name, logger, step):
    pwc('eval starts')
    ac = PPOAC(env.state_shape,
                env.action_dim,
                env.is_action_discrete,
                env.n_envs, 
                'ac')
    pwc('model is constructed')
    ac.set_weights(weights)
    i = 0
    while i < 10:
        i += env.n_envs
        state = env.reset()
        for j in range(env.max_episode_steps):
            print(j)
            action = ac.det_action(tf.convert_to_tensor(state, tf.float32))
            state, _, done, _ = env.step(action.numpy())

            if np.all(done):
                break
        logger.store(score=env.get_score(), epslen=env.get_epslen())

    stats = dict(
            model_name=f'{name}',
            timing='Eval',
            steps=f'{step}',
            score=logger.get('score', std=True), 
            epslen=logger.get('epslen', std=True)
    )
    [logger.log_tabular(k, v) for k, v in stats.items()]
    logger.dump_tabular()
