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
from torch import nn
import onnx2torch
import onnxruntime


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.utils import configure_gpu
from tools.display import print_dict_info
from tools import pkg
from th.nn.mlp import MLP
from th.nn.utils import get_activation, init_linear
from run.utils import search_for_config




class Categorical(nn.Module):
  def __init__(
    self, 
    num_inputs, 
    num_outputs, 
    out_w_init='orthogonal', 
    out_b_init='zeros', 
    out_scale=0.01
  ):
    super().__init__()
    self.linear = nn.Linear(num_inputs, num_outputs)
    init_linear(self.linear, out_w_init, out_b_init, out_scale)

  def forward(self, x, action_mask=None):
    x = self.linear(x)
    if action_mask is not None:
      x[action_mask == 0] = -1e10
    return x
    # return torch.distributions.Categorical(logits=x)


class DiagGaussian(nn.Module):
  def __init__(
    self, 
    num_inputs, 
    num_outputs, 
    out_w_init='orthogonal', 
    out_b_init='zeros', 
    out_scale=0.01, 
    sigmoid_scale=True, 
    std_x_coef=1., 
    std_y_coef=.5, 
    init_std=.2
  ):
    super().__init__()
    self.linear = nn.Linear(num_inputs, num_outputs)
    init_linear(self.linear, out_w_init, out_b_init, out_scale)
    self.sigmoid_scale = sigmoid_scale
    self.std_x_coef = std_x_coef
    self.std_y_coef = std_y_coef
    if sigmoid_scale:
      self.logstd = nn.Parameter(std_x_coef + torch.zeros(num_outputs))
    else:
      self.logstd = nn.Parameter(np.log(init_std) + torch.zeros(num_outputs))

  def forward(self, x):
    mean = self.linear(x)
    if self.sigmoid_scale:
      scale = torch.sigmoid(self.logstd / self.std_x_coef) * self.std_y_coef
    else:
      scale = torch.exp(self.logstd)
    return mean, scale
  

class TorchPolicy(nn.Module):
  def __init__(
    self, 
    input_dim, 
    is_action_discrete, 
    action_dim, 
    out_act=None, 
    init_std=.2, 
    sigmoid_scale=True, 
    std_x_coef=1., 
    std_y_coef=.5, 
    use_action_mask={'action': False}, 
    use_feature_norm=False, 
    out_w_init='orthogonal', 
    out_b_init='zeros', 
    out_scale=.01, 
    **config
  ):
    super().__init__()
    self.config = config
    self.action_dim = action_dim
    self.is_action_discrete = is_action_discrete

    self.out_act = out_act
    self.init_std = init_std
    self.sigmoid_scale = sigmoid_scale
    self.std_x_coef = std_x_coef
    self.std_y_coef = std_y_coef
    self.use_action_mask = use_action_mask
    self.use_feature_norm = use_feature_norm
    if self.use_feature_norm:
      self.pre_ln = nn.LayerNorm(input_dim)
    self.mlp = MLP(input_dim, **self.config)
    self.head_cont = None
    self.head_disc = None
    for k in action_dim:
      if is_action_discrete[k]:
        self.head_disc = Categorical(
          self.config['rnn_units'], action_dim[k], 
          out_w_init, out_b_init, out_scale)
      else:
        self.head_cont = DiagGaussian(
          self.config['rnn_units'], action_dim[k], 
          out_w_init, out_b_init, out_scale, 
          sigmoid_scale=sigmoid_scale, std_x_coef=std_x_coef, 
          std_y_coef=std_y_coef, init_std=init_std)

  def forward(self, x, reset=None, state=None):
    if self.use_feature_norm:
      x = self.pre_ln(x)
    x = self.mlp(x, reset, state)
    if isinstance(x, tuple):
      assert len(x) == 2, x
      x, state = x
    
    outs = {}
    for name in self.action_dim:
      if self.is_action_discrete[name]:
        d = self.head_disc(x)
      else:
        d = self.head_cont(x)
      outs[name] = d
    if state is None:
      return outs['action']
    else:
      return outs['action'], state


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
  configure_gpu(None)

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
  print_dict_info(params)
  
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
  model_config = config.model.policy.copy()
  model_config.pop('nn_id')
  Policy = pkg.import_module('elements.nn', algo=algo).Policy
  def create_policy(*args, **kwargs):
    policy_config = model_config.copy()
    policy_config.pop('rnn_init', None)
    policy_config['use_action_mask'] = env_stats.use_action_mask[config.aid]
    policy = Policy(
      env_stats.is_action_discrete[config.aid], 
      env_stats.action_dim[config.aid], 
      **policy_config, name='policy')
    return policy(*args, **kwargs)
  init, apply = hk.without_apply_rng(hk.transform(create_policy))
  rng = jax.random.PRNGKey(42)
  if model_config.rnn_type:
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
  print('jax out', jax_out)

  # # convert model to tf.function
  # tf_params = tf.nest.map_structure(tf.Variable, params)
  # if model_config.rnn_type:
  #   tf_policy = lambda x, reset, state, action_mask: \
  #     jax2tf.convert(policy, enable_xla=False)(tf_params, x, reset, state, action_mask)
  # else:
  #   tf_policy = lambda x: jax2tf.convert(policy, enable_xla=False)(tf_params, x)
  # tf_policy = tf.function(tf_policy, jit_compile=True, autograph=False)
  # tf_data = tf.Variable(obs, dtype=tf.float32, name='x')

  # if not os.path.exists(model_path):
  #   os.mkdir(model_path)
  # # save model as tf SavedModel
  # # tf_module = tf.Module()
  # # tf_model_path = f'{model_path}/tf_model'
  # # tf_module.vars = params
  # # tf_module.policy = tf_policy
  # # signature = tf_policy.get_concrete_function(
  # #     tf.TensorSpec(obs.shape, tf.float32))
  # # tf.saved_model.save(
  # #   tf_module, tf_model_path, 
  # #   signatures=signature
  # # )
  # # module = tf.saved_model.load(tf_model_path)

  # # save model as an onnx model
  # onnx_model_path = f'{model_path}/onnx_model.onnx'
  # if model_config.rnn_type:
  #   input_signature = [
  #     tf.TensorSpec(obs.shape, tf.float32, name='x'), 
  #     tf.TensorSpec(reset.shape, tf.float32, name='reset'), 
  #     type(state)(*[tf.TensorSpec(v.shape, tf.float32, name=k) for k, v in state._asdict().items()]), 
  #     {k: tf.TensorSpec(v.shape, tf.float32, name=k) for k, v in action_mask.items()}, 
  #   ]
  # else:
  #   input_signature=[tf.TensorSpec(obs.shape, tf.float32, name='x')]
  # print(input_signature)
  # onnx_model, _ = tf2onnx.convert.from_function(
  #   tf_policy, input_signature=input_signature)
  # onnx.save(onnx_model, onnx_model_path)
  # session = onnxruntime.InferenceSession(onnx_model_path)
  # onnx_out = session.run(None, {'x': obs})
  # print('onnx out', onnx_out)
  # np.testing.assert_allclose(jax_out['action'], onnx_out[0], rtol=1e-3, atol=1e-3)

  # convert onnx to torch
  # torch_model = onnx2torch.convert(onnx_model_path)
  x = torch.from_numpy(obs)
  th_policy = TorchPolicy(
    x.shape[-1], 
    env_stats.is_action_discrete[config.aid], 
    env_stats.action_dim[config.aid], 
    **model_config
  )
  print(th_policy)
  to_np = lambda x: np.array(x)
  swapaxes = lambda x: np.swapaxes(x, 0, 1)
  jax2np = lambda x: swapaxes(to_np(x)) if len(x.shape) == 2 else to_np(x)
  th_params = jax.tree_map(jax2np, params)
  with torch.no_grad():
    if th_policy.use_feature_norm:
      th_policy.pre_ln.weight.copy_(torch.tensor(th_params['policy/layer_norm']['scale']))
      th_policy.pre_ln.bias.copy_(torch.tensor(th_params['policy/layer_norm']['offset']))
    for i, layers in enumerate(th_policy.mlp.layers):
      suffix = '' if i == 0 else f'_{i}'
      layers[0].weight.copy_(torch.tensor(th_params[f'policy/mlp/linear'+suffix]['w']))
      layers[0].bias.copy_(torch.tensor(th_params[f'policy/mlp/linear'+suffix]['b']))
      layers[2].weight.copy_(torch.tensor(th_params[f'policy/mlp/layer_norm'+suffix]['scale']))
      layers[2].bias.copy_(torch.tensor(th_params[f'policy/mlp/layer_norm'+suffix]['offset']))
    th_policy.head_disc.linear.weight.copy_(torch.tensor(th_params['policy/head_action']['w']))
    th_policy.head_disc.linear.bias.copy_(torch.tensor(th_params['policy/head_action']['b']))
  # traced_model = torch.jit.trace(torch_model, x)
  torch_model_path = f'{model_path}/torch_model.pt'
  scripted_model = torch.jit.trace(th_policy, x)
  scripted_model.save(torch_model_path)
  torch_out = th_policy(x).detach().numpy()
  # torch.save(torch_model, torch_model_path)
  print('torch out', torch_out)

  np.testing.assert_allclose(jax_out['action'], torch_out, rtol=1e-3, atol=1e-3)
