import warnings
warnings.filterwarnings("ignore")

import argparse
import os, sys
import cloudpickle
os.environ['XLA_FLAGS'] = "--xla_gpu_force_compilation_parallelism=1"

import random
import numpy as np
import jax
from jax.experimental import jax2tf
import haiku as hk
import tensorflow as tf
import tf2onnx
import onnx
import torch
import onnx2torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools import pkg
from run.utils import search_for_config

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    'directory',
    type=str,
    help='directory where checkpoints and "config.yaml" exist')
  parser.add_argument(
    '--out_path', 
    type=str, 
    help='directory where the new models are saved', 
    default='./saved_model'
  )
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  args = parse_args()
  model_path = args.out_path
  if not os.path.isdir(model_path):
    os.mkdir(model_path)

  # load respective config
  config = search_for_config(args.directory)
  env_stats = config.env_stats

  # load parameters
  for policy in ['policy', 'policies']:
    policy_ckpt = os.path.join(
      config.root_dir, config.model_name, 'params', 'model', f'{policy}.pkl')
    if os.path.exists(policy_ckpt):
      break
  with open(policy_ckpt, 'rb') as f:
    params = cloudpickle.load(f)
  
  # load ancillary data
  anc_ckpt = os.path.join(config.root_dir, config.model_name, 'params', 'ancillary.pkl')
  obs_anc = None
  if os.path.exists(anc_ckpt):
    with open(anc_ckpt, 'rb') as f:
      anc = cloudpickle.load(f)
    idx = np.random.randint(len(anc.obs))
    print('ancillary data', anc)
    obs_anc = anc.obs[idx]
  if obs_anc:
    obs_anc = obs_anc['obs']
    mean = anc.mean[0]
    std = np.sqrt(anc.var)[0]
    obs_dim = mean.shape[-1]
  else:
    obs_dim = config.env_stats.obs_shape[0].obs[0]
    mean, std = np.zeros((obs_dim,)), np.ones((obs_dim,))
  np.savetxt(f'{model_path}/anc_mean.txt', mean)
  np.savetxt(f'{model_path}/anc_std.txt', np.sqrt(std))

  algo_name = config.algorithm
  env_name = config.env['env_name']
  # construct a fake obs
  obs = np.arange(obs_dim).reshape(1, 1, 1, obs_dim).astype(np.float32) / obs_dim
  reset = np.ones((1, 1, 1))
  action_dim = env_stats.action_dim[config.aid]
  action_mask = {k: np.random.randint(0, 2, (1, 1, 1, v)) for k, v in action_dim.items()}

  # build the policy model
  algo = algo_name.split('-')[-1]
  Policy = pkg.import_module('elements.nn', algo=algo).Policy
  def create_policy(*args, **kwargs):
    policy_config = config.model.policy.copy()
    policy_config.pop('nn_id')
    policy_config['use_action_mask'] = env_stats.use_action_mask[config.aid]
    policy = Policy(
      env_stats.is_action_discrete[config.aid], 
      env_stats.action_dim[config.aid], 
      **policy_config, name='policy')
    return policy(*args, **kwargs)
  init, apply = hk.without_apply_rng(hk.transform(create_policy))
  rng = jax.random.PRNGKey(42)
  if config.model.policy.rnn_type:
    init_params = init(rng, obs, reset, None, action_mask=action_mask)
    policy = lambda p, x, reset, state, action_mask: apply(p, x, reset, state, action_mask=action_mask)
    if isinstance(params, list):
      params = params[idx]
    jax_out, state = policy(params, obs, reset, None, action_mask=action_mask)
  else:
    init_params = init(rng, obs, no_state_return=True)
    policy = lambda p, x: apply(p, x, no_state_return=True)
    if isinstance(params, list):
      params = params[idx]
    jax_out = policy(params, obs)

  # convert model to tf.function
  params = tf.nest.map_structure(tf.Variable, params)
  tf_policy = lambda x, reset, state, action_mask: \
    jax2tf.convert(policy, enable_xla=False)(params, x, reset, state, action_mask)
  tf_policy = tf.function(tf_policy, jit_compile=True, autograph=False)
  tf_data = tf.Variable(obs, dtype=tf.float32, name='x')

  if not os.path.exists(model_path):
    os.mkdir(model_path)
  # save model as tf SavedModel
  # tf_module = tf.Module()
  # tf_model_path = f'{model_path}/tf_model'
  # tf_module.vars = params
  # tf_module.policy = tf_policy
  # signature = tf_policy.get_concrete_function(
  #     tf.TensorSpec(obs.shape, tf.float32))
  # tf.saved_model.save(
  #   tf_module, tf_model_path, 
  #   signatures=signature
  # )
  # module = tf.saved_model.load(tf_model_path)

  # save model as an onnx model
  onnx_model_path = f'{model_path}/onnx_model.onnx'
  if config.model.policy.rnn_type:
    input_signature = [
      tf.TensorSpec(obs.shape, tf.float32, name='x'), 
      tf.TensorSpec(reset.shape, tf.float32, name='reset'), 
      type(state)(*[tf.TensorSpec(v.shape, tf.float32, name=k) for k, v in state._asdict().items()]), 
      {k: tf.TensorSpec(v.shape, tf.float32, name=k) for k, v in action_mask.items()}, 
    ]
  else:
    input_signature=[tf.TensorSpec(obs.shape, tf.float32, name='x')]
  print(input_signature)
  onnx_model, _ = tf2onnx.convert.from_function(
    tf_policy, input_signature=input_signature)
  onnx.save(onnx_model, onnx_model_path)
  
  # convert onnx to torch
  torch_model = onnx2torch.convert(onnx_model_path)
  x = torch.from_numpy(obs)
  traced_model = torch.jit.trace(torch_model, x)
  torch_model_path = f'{model_path}/torch_model.pt'
  traced_model.save(torch_model_path)
  torch_out = traced_model(x)
  # torch.save(torch_model, torch_model_path)
  print('jax out', jax_out)
  print('torch out', torch_out.numpy())

  np.testing.assert_allclose(jax_out['action'], torch_out, rtol=1e-3, atol=1e-3)
