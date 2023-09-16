import argparse
import os
import string
import cloudpickle
import numpy as np
import pandas as pd
import jax
from jax import lax, nn, random
import jax.numpy as jnp
import optax
import chex
import matplotlib.pyplot as plt
import seaborn as sns
import ray


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--lr', '-lr', 
            type=float, 
            default=1e-4)
  parser.add_argument('-iteration', '-i', 
            type=int, 
            default=1000)
  parser.add_argument('-epochs', '-e', 
            type=int, 
            default=1)
  parser.add_argument('--horizon', '-H',
            type=int, 
            default=1)
  parser.add_argument('--state_size', 
            type=int, 
            default=3)
  parser.add_argument('--action_dims', 
            type=int, 
            nargs='*',
            default=[3, 3])
  parser.add_argument('--gamma', 
            type=float, 
            default=.99)
  parser.add_argument('--seed', '-s', 
            type=int, 
            default=1)
  parser.add_argument('--dir_path', '-d', 
            type=str, 
            default='standalone/results')
  parser.add_argument('--name', '-n', 
            type=str, 
            default=None)
  parser.add_argument('--reward_set', '-rs', 
            type=int, 
            nargs='*', 
            default=[-5, 0, 3, 1])
  parser.add_argument('--reward_dist', '-rd', 
            type=float, 
            nargs='*', 
            default=[.2, .5, 1, .2])
  parser.add_argument('--n_lka_steps', '-lka', 
            type=int,  
            default=1)

  args = parser.parse_args()

  return args


""" Algorithm """
def prob_from_logits(logits):
  prob = jax.tree_util.tree_map(
    lambda x: nn.softmax(x, -1), logits)
  return prob


def ratio(pi, mu):
  return pi / mu


def joint_prob(p1, p2):
  """ Compute stateless joint probability distribution
  Params:
    p1 shape1
    p2 shape2
  Return:
    p (*shape1, *shape2)
  """
  p1_dims = string.ascii_lowercase[:p1.ndim]
  p2_dims = string.ascii_lowercase[-p2.ndim:]
  p = jnp.einsum(f'{p1_dims},{p2_dims}->{p1_dims}{p2_dims}', p1, p2)

  return p


def vectorized_joint_prob(p1, p2):
  vjp = jax.vmap(joint_prob)
  p = vjp(p1, p2)
  chex.assert_trees_all_close(jnp.sum(p1, list(range(1, p1.ndim))), 1, rtol=1e-4, atol=1e-4)
  chex.assert_trees_all_close(jnp.sum(p2, list(range(1, p2.ndim))), 1, rtol=1e-4, atol=1e-4)
  chex.assert_trees_all_close(jnp.sum(p, list(range(1, p.ndim))), 1, rtol=1e-4, atol=1e-4)

  return p


def joint_prob_from_marginals(ps):
  jp = ps[0]
  for p in ps[1:]:
    jp = vectorized_joint_prob(jp, p)
  return jp


def exp_next_values(transition, next_value, gamma):
  """ Compute E_{s'\sim transition}[V^\mu(s')], 
    the expectation of the next values
  Params:
    transition (a, s)
    next_value (s, ): V^\mu(s')
  Return:
    value (a):  E_{s'\sim transition}[V^\mu(s')]
  """
  return gamma * jnp.sum(transition * next_value, -1)


def q_value(reward, transition, next_value, gamma):
  """ Compute Q^\mu(s, a) for state s associated to reward
  Params:
    mu (a): policy
    transition (a, s)
    reward (a)
    next_value (s, ): V^\mu(s')
  """
  q = reward + exp_next_values(transition, next_value, gamma)
  chex.assert_rank([q], 1)
  return q


def value_from_q(mu, q):
  """ Compute V^\mu(s) for state s associated to reward
  """
  v = jnp.sum(mu * q, -1)
  return v


def value(mu, reward, transition, next_value, gamma):
  """ Compute V^\mu(s) for state s associated to reward
  Params:
    mu (a): policy
    reward (a)
    transition (a, s)
    next_value (s, ): V^\mu(s')
  """
  q = q_value(reward, transition, next_value, gamma)
  v = value_from_q(mu, q)
  return v


def vectorized_value(mu, reward, transition, next_value, gamma):
  vv = jax.vmap(value, (0, 0, 0, None, None))
  v = vv(mu, reward, transition, next_value, gamma)
  return v


def advantage(joint_mu, reward, transition, next_value, gamma):
  """ Compute A_\mu(s) for state s associated to reward
  Params:
    joint_mu (a^1, ..., a^n): behavior policy
    reward (a, )
    transition (a, s)
    next_value (s, ): V^\mu(s')
  Return:
    adv (a, ): advantage
  """
  q = q_value(reward, transition, next_value, gamma)
  joint_mu = joint_mu.reshape(-1)
  v = value_from_q(joint_mu, q)
  adv = q - v
  chex.assert_rank([v], 0)
  chex.assert_rank([adv], 1)
  return adv, v


def vectorized_advantage(
    joint_mu, reward, transition, next_value, gamma):
  va = jax.vmap(advantage, (0, 0, 0, None, None))
  adv, v = va(joint_mu, reward, transition, next_value, gamma)
  return adv, v


def weighted_advantage(joint_mu, teammate_ratio, adv):
  """ Compute the i-th agent's advantage weighted 
    by the joint behavior policy and teammate ratio
  Params:
    joint_mu: (a^n, ..., a^1): \sum_{i=1}^{n} pi_i / mu_i
    teammate_ratio (a^{i-1}, ..., a^{1}): \sum_{j=1}^{i-1} pi_j / mu_j
    adv (a^n, ..., a^1): advantages
  
  """
  i = teammate_ratio.ndim
  _i = [-1 - j for j in range(adv.ndim) if j != i]
  adv = jnp.sum(joint_mu * teammate_ratio * adv, axis=_i)
  return adv


def vectorized_weighted_advantage(joint_mu, teammate_ratio, adv):
  vwa = jax.vmap(weighted_advantage)
  adv = vwa(joint_mu, teammate_ratio, adv)
  return adv


def ppo_loss(
  advantage, 
  pi, 
  mu, 
  clip_range, 
  gamma, 
):
  neg_adv = -advantage
  ratio_ = ratio(pi, mu)
  pg_loss = neg_adv * ratio_
  if clip_range is None:
    max_A = jnp.max(jnp.abs(advantage))
    tv = jnp.max(jnp.sum(pi - mu, -1), 0)
    loss = pg_loss - (4 * max_A * gamma * tv**2) / (1 - gamma)**2
  else:
    clipped_loss = neg_adv * jnp.clip(ratio_, 1. - clip_range, 1. + clip_range)
    loss = jnp.maximum(pg_loss, clipped_loss)
  
  return loss


def happo_loss_i(pi_logits_i, mu_i, joint_mu, teammate_ratio, adv, gamma):
  """ Compute the HAPPO loss function for agent i: 
  E_\mu[(ratio_i - 1) * teammate_ratio * adv]. 
  NOTE: 
    1. We assume a permutation has been performed 
    on teammatre_ratios, joint_mu, adv.
    2. For convenience, we assume the agents updated
    before i are in the higher dimensions: see shapes
    in Params
  Params:
    pi_logits_i (s, a^i): learning policy logits
    mu_i (s, a^i): behavior policy
    joint_mu: (s, a^n, ..., a^1): \sum_{i=1}^{n} pi_i / mu_i
    teammate_ratio (s, a^{i-1}, ..., a^{1}): \sum_{j=1}^{i-1} pi_j / mu_j
    adv (s, a^n, ..., a^1): advantages
  Returnn:
    loss
  """
  assert teammate_ratio.ndim > 0, teammate_ratio.ndim
  adv = vectorized_weighted_advantage(joint_mu, teammate_ratio, adv)
  pi_i = nn.softmax(pi_logits_i, -1)
  chex.assert_rank([pi_i, mu_i, adv], 2)
  chex.assert_equal_shape([pi_i, mu_i, adv])
  loss = ppo_loss(adv, pi_i, mu_i, None, gamma)
  loss = jnp.sum(loss)

  return loss


def happo_optimize_i(pi_logits, mu, joint_mu, teammate_ratio, adv, opt, state, gamma):
  grads = jax.grad(happo_loss_i)(
    pi_logits, mu, joint_mu, teammate_ratio, adv, gamma)
  updates, state = opt.update(grads, state)
  pi_logits = optax.apply_updates(pi_logits, updates)
  
  return pi_logits, state



jit_happo_optimize_i = jax.jit(happo_optimize_i, static_argnames='opt')


def update_teammate_ratio(teammate_ratio, pi_logits_i, mu_i):
  pi = nn.softmax(pi_logits_i)
  ratio_i = pi / mu_i
  tr_dims = string.ascii_lowercase[:teammate_ratio.ndim-1]
  teammate_ratio = jnp.einsum(
    f'sz,s{tr_dims}->sz{tr_dims}', 
    ratio_i, teammate_ratio
  )
  return teammate_ratio


def happo_optimize(rng, pi_logits, mu, joint_mu, adv, opts, states, gamma):
  """ Perform one-step HAPG optimization 
  Params:
    transition (s, a, s)
    pi_logits [(s, a^1), ..., (s, a^n)]: learning policy logits
    mu [(s, a^1), ..., (s, a^n)]: behavior policy
    joint_mu (s, a^1, ..., a^n): joint behavior policy
    adv (s, a^1, ..., a^n): advantage 
  """
  n_agents = len(pi_logits)
  assert len(pi_logits) == n_agents, pi_logits
  assert len(mu) == n_agents, mu
  
  aids = random.permutation(rng, jnp.arange(n_agents))
  new_axes = jnp.concatenate([jnp.zeros(1, jnp.int32), aids+1])
  adv = jnp.transpose(adv, new_axes)
  joint_mu = jnp.transpose(joint_mu, new_axes)
  chex.assert_equal_shape([adv, joint_mu])

  teammate_ratio = jnp.ones(adv.shape[0])
  idx = n_agents
  for i in reversed(aids):
    assert pi_logits[i].shape[-1] == joint_mu.shape[idx], (idx, pi_logits[i].shape, joint_mu.shape)
    pi_logits[i], states[i] = jit_happo_optimize_i(
      pi_logits[i], mu[i], joint_mu, teammate_ratio, adv, 
      opts[i], states[i], gamma, 
    )
    teammate_ratio = update_teammate_ratio(
      teammate_ratio, pi_logits[i], mu[i]
    )
    idx -= 1

  return pi_logits, states


def happo_train_t(
  rng, 
  pi_logits, 
  mu, 
  joint_mu, 
  adv, 
  gamma,
  opt, 
  state, 
):
  # joint_mu = joint_prob_from_marginals(mu)
  # adv, v = vectorized_advantage(
  #   joint_mu, reward, transition, next_value, gamma)
  adv = adv.reshape(joint_mu.shape)
  pi_logits, state = happo_optimize(
    rng, pi_logits, mu, joint_mu, adv, opt, state, gamma)

  return pi_logits, state


def happo_train(
  rng, 
  pi_logits, 
  mu, 
  reward, 
  transition, 
  gamma,
  opts, 
  states, 
  horizon, 
  epochs, 
):
  values = [
    jnp.zeros(transition.shape[0])
    for _ in range(horizon+1)
  ]
  advs = [
    jnp.zeros(transition.shape[:2])
    for _ in range(horizon)
  ]
  joint_mus = [
    jnp.zeros(transition.shape[:2])
    for _ in range(horizon)
  ]
  for t in reversed(range(horizon)):
    joint_mu = joint_prob_from_marginals(mu[t])
    adv, v = vectorized_advantage(
      joint_mu, reward[t], transition, values[t+1], gamma)
    joint_mus[t] = joint_mu
    advs[t] = adv
    values[t] = v

  for _ in range(epochs):
    for t in reversed(range(horizon)):
      rng, train_rng = random.split(rng)
      pi_logits[t], states[t] = happo_train_t(
        rng=train_rng, 
        pi_logits=pi_logits[t], 
        mu=mu[t], 
        joint_mu=joint_mus[t], 
        adv=advs[t], 
        gamma=gamma, 
        opt=opts[t], 
        state=states[t], 
      )

  return pi_logits, states


""" Initialization """
def build_name(args):
  if args.name is None:
    name = __file__.split('.')[0]
  else:
    name = args.name
  name = '-'.join([
    name, 
    f'horizon={args.horizon}', 
    f'epochs={args.epochs}', 
    f'lr={args.lr}', 
    f'state_size={args.state_size}', 
    f'action_dims={args.action_dims}', 
  ])

  return name


def build_initial_policy(horizon, state_size, action_dims):
  pi_logits = [
    [jnp.zeros((state_size, action_dim)) 
    for action_dim in action_dims] 
    for _ in range(horizon)
  ]

  return pi_logits


def build_optimizers(lr, theta):
  opts = jax.tree_util.tree_map(lambda _: optax.sgd(lr), theta)
  states = jax.tree_util.tree_map(lambda pl, opt: opt.init(pl), theta, opts)

  return opts, states


def build_dynamics(
  horizon, state_size, action_dims, seed, 
  reward_set, reward_dist
):
  rng = random.PRNGKey(seed)
  rngs = random.split(rng, horizon+1)
  reward_set = jnp.array(reward_set)
  reward_dist = jnp.array(reward_dist)
  joint_action_dim = np.prod(action_dims)
  reward = [
    random.choice(
      rngs[h], 
      reward_set, 
      shape=[state_size, joint_action_dim], 
      p=reward_dist
    ) for h in range(horizon)
  ]
  # initial state distributiion
  rngs = random.split(rngs[-1], 2)
  rho = random.uniform(rngs[0], (state_size, ))
  rho = rho / jnp.sum(rho)
  
  trans_shape = (state_size, joint_action_dim, state_size)
  transition = random.uniform(rngs[1], trans_shape, minval=1, maxval=10)
  mask_prob = jnp.array([.8, .2])
  mask = random.choice(rngs[2], jnp.arange(2), shape=trans_shape, p=mask_prob)
  masked_trans = transition * mask
  transition = jnp.where(jnp.sum(masked_trans, -1, keepdims=True) == 0, transition, masked_trans)
  transition = transition / jnp.sum(transition, -1, keepdims=True)
  chex.assert_trees_all_close(jnp.sum(transition, -1), 1, rtol=1e-4, atol=1e-4)

  return reward, rho, transition


def print_dynamics(reward, rho, transition, action_dims):
  print('Reward:')
  for t, r in enumerate(reward):
    print('time', t)
    for s, r in enumerate(r):
      print('state', s)
      print(r.reshape(action_dims))
  
  print('Initial State Distribution')
  print(rho)
  print('Transition')
  for s, ans in enumerate(transition):
    print('state', s)
    print(ans)


""" Evaluation """
def evaluate(pi, reward, rho, transition):
  horizon = len(reward)
  vs = [jnp.zeros(reward[0].shape[0])]
  for t in reversed(range(horizon)):
    pi_t = pi[t]
    reward_t = reward[t]
    joint_pi = joint_prob_from_marginals(pi_t)
    joint_pi = joint_pi.reshape(reward_t.shape)
    vs.append(vectorized_value(joint_pi, reward_t, transition, vs[-1], 1))
  vs = vs[::-1]
  score = jnp.mean(vs[0] * rho)
  
  return score


# def plot(x, y, name):
#   plt.xlabel('step')
#   plt.ylabel('score')
#   plt.plot(x, y)
#   path = os.path.abspath(f'{name}.png')
#   plt.savefig(path)
#   print(f'File saved at "{path}"')


def process_data(data, dir_path, name):
  data = pd.DataFrame.from_dict(data=data)
  data.to_csv(f'{dir_path}/{name}.txt')
  return data


def plot(data, dir_path, name):
  fig = plt.figure(figsize=(20, 10))
  fig.tight_layout(pad=2)
  ax = fig.add_subplot()
  sns.set(style="whitegrid", font_scale=1.5)
  sns.set_palette('Set2') # or husl
  sns.lineplot(x='steps', y='score', 
    ax=ax, data=data, dashes=False, linewidth=3, hue='legend')
  ax.grid(True, alpha=0.8, linestyle=':')
  ax.legend(loc='best').set_draggable(True)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.set_title(name)
  fig_path = '/'.join([dir_path, f'{name}.png'])
  fig.savefig(fig_path)
  print(f'File saved at "{fig_path}"')


""" Training """
def train(
  args, 
  pi_logits, 
  reward, 
  rho, 
  transition, 
  opts, 
  states, 
  points=100, 
):
  interval = args.iteration // points
  print('Interval:', interval)
  rng = random.PRNGKey(args.seed)
  # print(f'Initial PRNG for seed={seed}:', rng)
  mu = prob_from_logits(pi_logits)
  steps = [0]
  scores = [evaluate(mu, reward, rho, transition)]
  print(f'{args.name} Iteration {0}:\t{scores[0]}')

  for i in range(1, args.iteration+1):
    pi_logits, states = happo_train(
      rng=rng, 
      pi_logits=pi_logits, 
      mu=mu, 
      reward=reward, 
      transition=transition, 
      gamma=args.gamma, 
      opts=opts, 
      states=states, 
      horizon=args.horizon, 
      epochs=args.epochs, 
    )

    mu = prob_from_logits(pi_logits)
    if i % interval == 0:
      score = evaluate(mu, reward, rho, transition)
      steps.append(i)
      scores.append(score)
      print(f'{args.name} Iteration {i}:\t{score}')
  steps = np.array(steps)
  scores = np.array(scores)

  return steps, scores


def build_and_train(
  args, 
  points=100, 
  build_initial_policy=build_initial_policy, 
  build_optimizers=build_optimizers, 
  build_dynamics=build_dynamics, 
  train=train
):
  horizon = args.horizon
  state_size = args.state_size
  action_dims = args.action_dims
  pi_logits = build_initial_policy(
    horizon, state_size, action_dims
  )
  opts, states = build_optimizers(args.lr, pi_logits)

  reward, rho, transition = build_dynamics(
    horizon, state_size, action_dims, args.seed, 
    args.reward_set, args.reward_dist
  )

  # print_dynamics(reward, rho, transition, action_dims)

  steps, scores = train(
    args, 
    pi_logits, 
    reward, 
    rho, 
    transition, 
    opts, 
    states, 
    points=points, 
  )

  return steps, scores


def main(
  args, 
  build_initial_policy=build_initial_policy, 
  build_optimizers=build_optimizers, 
  build_dynamics=build_dynamics, 
  train=train
):
  horizon = args.horizon
  epochs = args.epochs
  state_size = args.state_size
  action_dims = args.action_dims
  n_agents = len(action_dims)
  print('Horizon:', horizon)
  print('Epochs:', epochs)
  print('#Agents:', n_agents)
  print('State size:', state_size)
  print('Action dimensions:', action_dims)

  data = {}
  points = 100
  max_seed = args.seed
  if max_seed == 0:
    steps, scores = build_and_train(
      args, 
      points=points, 
      build_initial_policy=build_initial_policy, 
      build_optimizers=build_optimizers, 
      build_dynamics=build_dynamics, 
      train=train
    )
  else:
    processes = []
    ray_bt = ray.remote(build_and_train)
    for seed in range(max_seed):
      args.seed = seed
      p = ray_bt.remote(
        args, 
        points=points, 
        build_initial_policy=build_initial_policy, 
        build_optimizers=build_optimizers, 
        build_dynamics=build_dynamics, 
        train=train
      )
      processes.append(p)

    results = ray.get(processes)
    steps, scores = list(zip(*results))
    steps = np.concatenate(steps)
    scores = np.concatenate(scores)
  data['steps'] = steps
  data['score'] = scores
  data['legend'] = [args.name] * data['steps'].shape[0]

  return data


if __name__ == '__main__':
  ray.init()

  args = parse_args()
  args.algo = 'happo'
  args.name = 'happo'
  args.name = build_name(args)
  data = main(args)
  dir_path = args.dir_path
  dir_path = os.path.abspath(dir_path)
  if not os.path.isdir(dir_path):
    os.mkdir(dir_path)

  data = process_data(data, dir_path, args.name)
  plot(data, dir_path, args.name)
  
  ray.shutdown()
