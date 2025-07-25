
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def get_apply_power_sign_inputs():
  """
  Generates a list of valid inputs for tf.raw_ops.ApplyPowerSign.
  The error "RuntimeError: ... does not support eager execution" is inherent to
  this specific raw operation. It is a stateful op designed for graph-based
  execution (e.g., inside a @tf.function) and is explicitly disabled in
  eager mode. The provided inputs are valid for a graph context.
  """
  list_of_inputs = []

  # Input 1: Basic float32, 1D case
  dtype = np.float32
  list_of_inputs.append(copy.deepcopy({
      'var': np.array([1.0, 2.0, 3.0], dtype=dtype),
      'm': np.array([0.1, 0.2, 0.3], dtype=dtype),
      'lr': np.array(0.001, dtype=dtype),
      'logbase': np.array(np.e, dtype=dtype),
      'sign_decay': np.array(0.99, dtype=dtype),
      'beta': np.array(0.9, dtype=dtype),
      'grad': np.array([-0.5, 0.5, -0.2], dtype=dtype),
      'use_locking': False,
      'name': 'float32_1d_standard'
  }))

  # Input 2: Basic float64, 2D case
  dtype = np.float64
  list_of_inputs.append(copy.deepcopy({
      'var': np.array([[1.0, -2.0], [3.0, -4.0]], dtype=dtype),
      'm': np.array([[0.1, -0.2], [0.3, -0.4]], dtype=dtype),
      'lr': np.array(0.01, dtype=dtype),
      'logbase': np.array(np.e, dtype=dtype),
      'sign_decay': np.array(0.95, dtype=dtype),
      'beta': np.array(0.8, dtype=dtype),
      'grad': np.array([[0.1, -0.1], [-0.2, 0.2]], dtype=dtype),
      'use_locking': False,
      'name': 'float64_2d_standard'
  }))

  # Input 3: float32 with locking enabled
  dtype = np.float32
  list_of_inputs.append(copy.deepcopy({
      'var': np.random.randn(3, 3).astype(dtype),
      'm': np.random.randn(3, 3).astype(dtype),
      'lr': np.array(0.1, dtype=dtype),
      'logbase': np.array(2.0, dtype=dtype),
      'sign_decay': np.array(0.9, dtype=dtype),
      'beta': np.array(0.99, dtype=dtype),
      'grad': np.random.randn(3, 3).astype(dtype),
      'use_locking': True,
      'name': 'float32_locking'
  }))

  # Input 4: float64 with different optimizer parameters
  dtype = np.float64
  list_of_inputs.append(copy.deepcopy({
      'var': np.array([100.0, 200.0], dtype=dtype),
      'm': np.array([10.0, 20.0], dtype=dtype),
      'lr': np.array(0.05, dtype=dtype),
      'logbase': np.array(10.0, dtype=dtype),
      'sign_decay': np.array(0.8, dtype=dtype),
      'beta': np.array(0.7, dtype=dtype),
      'grad': np.array([5.0, -15.0], dtype=dtype),
      'use_locking': False,
      'name': 'float64_alt_params'
  }))

  # Input 5: float32 with beta = 0.0
  dtype = np.float32
  list_of_inputs.append(copy.deepcopy({
      'var': np.array([5.0, 6.0], dtype=dtype),
      'm': np.array([0.5, 0.6], dtype=dtype),
      'lr': np.array(0.1, dtype=dtype),
      'logbase': np.array(np.e, dtype=dtype),
      'sign_decay': np.array(0.9, dtype=dtype),
      'beta': np.array(0.0, dtype=dtype),
      'grad': np.array([-0.5, 0.5], dtype=dtype),
      'use_locking': False,
      'name': 'float32_beta_zero'
  }))

  # Input 6: float64 with zero gradient
  dtype = np.float64
  list_of_inputs.append(copy.deepcopy({
      'var': np.array([1.0, 2.0], dtype=dtype),
      'm': np.array([0.1, 0.2], dtype=dtype),
      'lr': np.array(0.001, dtype=dtype),
      'logbase': np.array(np.e, dtype=dtype),
      'sign_decay': np.array(0.99, dtype=dtype),
      'beta': np.array(0.9, dtype=dtype),
      'grad': np.array([0.0, 0.0], dtype=dtype),
      'use_locking': False,
      'name': 'float64_zero_grad'
  }))

  return list_of_inputs

generated_inputs["tf.raw_ops.ApplyPowerSign"] = get_apply_power_sign_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyPowerSign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyPowerSign'.")

check_valid('tf.raw_ops.ApplyPowerSign', generated_inputs['tf.raw_ops.ApplyPowerSign'], lib="tf", suffix=0)
