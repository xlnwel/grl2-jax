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

  # load respective config
  config = search_for_config(args.directory)

  # load parameters
  for policy in ['policy', 'policies']:
    policy_ckpt = os.path.join(
      config.root_dir, config.model_name, 'params', 'model', f'{policy}.pkl')
    if os.path.exists(policy_ckpt):
      break
  with open(policy_ckpt, 'rb') as f:
    params = cloudpickle.load(f)
  anc_ckpt = os.path.join(config.root_dir, config.model_name, 'params', 'ancillary.pkl')
  with open(anc_ckpt, 'rb') as f:
    anc = cloudpickle.load(f)
  idx = np.random.randint(len(anc.obs))
  anc = anc.obs[idx]['obs']
  mean = anc.mean[0]
  std = np.sqrt(anc.var)[0]
  np.savetxt(f'{model_path}/anc_mean.txt', mean)
  np.savetxt(f'{model_path}/anc_std.txt', np.sqrt(std))
  print(f'mean={mean.shape}')
  print(f'std={std.shape}')

  algo_name = config.algorithm
  env_name = config.env['env_name']
  # construct a fake obs
  obs_dim = mean.shape[-1]
  obs = np.arange(115).reshape(1, obs_dim).astype(np.float32) / obs_dim

  # build the policy model
  algo = algo_name.split('-')[-1]
  Policy = pkg.import_module('elements.nn', algo=algo).Policy
  def create_policy(*args, **kwargs):
    policy_config = config.model.policy.copy()
    policy_config.pop('nn_id')
    policy = Policy(**policy_config, name='policy')
    return policy(*args, **kwargs)
  init, apply = hk.without_apply_rng(hk.transform(create_policy))
  rng = jax.random.PRNGKey(42)
  init_params = init(rng, obs, no_state_return=True)
  policy = lambda p, x: apply(p, x, no_state_return=True)
  if isinstance(params, list):
    params = params[idx]
  jax_out = policy(params, obs)

  # convert model to tf.function
  params = tf.nest.map_structure(tf.Variable, params)
  tf_policy = lambda x: jax2tf.convert(policy, enable_xla=False)(params, x)
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
  onnx_model, _ = tf2onnx.convert.from_function(
    tf_policy, input_signature=[tf.TensorSpec(obs.shape, tf.float32, name='x')])
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
