
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_SparseApplyAdadelta_inputs():
  """
  Generates a list of valid inputs for tf.raw_ops.SparseApplyAdadelta.
  The persistent error 'RuntimeError: sparse_apply_adadelta op does not support eager execution'
  indicates that this low-level operation is designed for TensorFlow's graph execution
  mode and lacks an implementation for the default eager mode. The inputs provided here
  are valid according to the API signature but will fail in an eager context. The function
  provides a variety of valid inputs focusing on standard float types.
  """
  list_of_inputs = []

  # Input 1: Basic float32, 2D
  dtype = np.float32
  list_of_inputs.append(copy.deepcopy({
      'var': np.arange(10, dtype=dtype).reshape(5, 2),
      'accum': np.ones((5, 2), dtype=dtype),
      'accum_update': np.zeros((5, 2), dtype=dtype),
      'lr': np.array(0.001, dtype=dtype),
      'rho': np.array(0.95, dtype=dtype),
      'epsilon': np.array(1e-8, dtype=dtype),
      'grad': np.random.rand(3, 2).astype(dtype),
      'indices': np.array([0, 2, 4], dtype=np.int32),
      'use_locking': False,
      'name': "s_a_adadelta_f32_2d"
  }))

  # Input 2: Basic float64, 2D
  dtype = np.float64
  list_of_inputs.append(copy.deepcopy({
      'var': np.arange(10, dtype=dtype).reshape(5, 2),
      'accum': np.ones((5, 2), dtype=dtype),
      'accum_update': np.zeros((5, 2), dtype=dtype),
      'lr': np.array(0.001, dtype=dtype),
      'rho': np.array(0.95, dtype=dtype),
      'epsilon': np.array(1e-7, dtype=dtype),
      'grad': np.random.rand(2, 2).astype(dtype),
      'indices': np.array([1, 3], dtype=np.int32),
      'use_locking': False,
      'name': "s_a_adadelta_f64_2d"
  }))

  # Input 3: float32, 1D var
  dtype = np.float32
  list_of_inputs.append(copy.deepcopy({
      'var': np.arange(10, dtype=dtype),
      'accum': np.ones(10, dtype=dtype),
      'accum_update': np.zeros(10, dtype=dtype),
      'lr': np.array(0.1, dtype=dtype),
      'rho': np.array(0.9, dtype=dtype),
      'epsilon': np.array(1e-8, dtype=dtype),
      'grad': np.array([0.1, 0.2, 0.3], dtype=dtype),
      'indices': np.array([1, 5, 9], dtype=np.int32),
      'use_locking': False,
      'name': "s_a_adadelta_f32_1d"
  }))

  # Input 4: use_locking=True
  dtype = np.float32
  list_of_inputs.append(copy.deepcopy({
      'var': np.arange(6, dtype=dtype).reshape(3, 2),
      'accum': np.ones((3, 2), dtype=dtype),
      'accum_update': np.zeros((3, 2), dtype=dtype),
      'lr': np.array(0.01, dtype=dtype),
      'rho': np.array(0.9, dtype=dtype),
      'epsilon': np.array(1e-7, dtype=dtype),
      'grad': np.random.rand(2, 2).astype(dtype),
      'indices': np.array([0, 2], dtype=np.int32),
      'use_locking': True,
      'name': "s_a_adadelta_locking"
  }))

  # Input 5: int64 indices
  dtype = np.float32
  list_of_inputs.append(copy.deepcopy({
      'var': np.arange(10, dtype=dtype).reshape(5, 2),
      'accum': np.ones((5, 2), dtype=dtype),
      'accum_update': np.zeros((5, 2), dtype=dtype),
      'lr': np.array(0.001, dtype=dtype),
      'rho': np.array(0.95, dtype=dtype),
      'epsilon': np.array(1e-8, dtype=dtype),
      'grad': np.random.rand(3, 2).astype(dtype),
      'indices': np.array([0, 2, 4], dtype=np.int64),
      'use_locking': False,
      'name': "s_a_adadelta_i64_indices"
  }))

  # Input 6: Single index update
  dtype = np.float32
  list_of_inputs.append(copy.deepcopy({
      'var': np.ones((5, 3), dtype=dtype),
      'accum': np.full((5, 3), 0.1, dtype=dtype),
      'accum_update': np.full((5, 3), 0.1, dtype=dtype),
      'lr': np.array(0.1, dtype=dtype),
      'rho': np.array(0.9, dtype=dtype),
      'epsilon': np.array(1e-7, dtype=dtype),
      'grad': np.random.rand(1, 3).astype(dtype),
      'indices': np.array([3], dtype=np.int32),
      'use_locking': False,
      'name': "s_a_adadelta_single_index"
  }))

  # Input 7: All zeros grad
  dtype = np.float32
  list_of_inputs.append(copy.deepcopy({
      'var': np.arange(10, dtype=dtype).reshape(5, 2),
      'accum': np.ones((5, 2), dtype=dtype),
      'accum_update': np.ones((5, 2), dtype=dtype),
      'lr': np.array(0.001, dtype=dtype),
      'rho': np.array(0.95, dtype=dtype),
      'epsilon': np.array(1e-8, dtype=dtype),
      'grad': np.zeros((3, 2), dtype=dtype),
      'indices': np.array([0, 2, 4], dtype=np.int32),
      'use_locking': False,
      'name': "s_a_adadelta_zero_grad"
  }))

  # Input 8: High epsilon
  dtype = np.float32
  list_of_inputs.append(copy.deepcopy({
      'var': np.arange(10, dtype=dtype).reshape(5, 2),
      'accum': np.ones((5, 2), dtype=dtype),
      'accum_update': np.zeros((5, 2), dtype=dtype),
      'lr': np.array(0.001, dtype=dtype),
      'rho': np.array(0.95, dtype=dtype),
      'epsilon': np.array(1.0, dtype=dtype),
      'grad': np.random.rand(3, 2).astype(dtype),
      'indices': np.array([0, 2, 4], dtype=np.int32),
      'use_locking': False,
      'name': "s_a_adadelta_high_epsilon"
  }))

  # Input 9: Negative values in var
  dtype = np.float32
  list_of_inputs.append(copy.deepcopy({
      'var': np.array([[-1.0, 2.0], [-3.0, 4.0], [-5.0, 6.0]], dtype=dtype),
      'accum': np.abs(np.random.randn(3, 2)).astype(dtype) + 1.0,
      'accum_update': np.abs(np.random.randn(3, 2)).astype(dtype) + 1.0,
      'lr': np.array(0.01, dtype=dtype),
      'rho': np.array(0.9, dtype=dtype),
      'epsilon': np.array(1e-7, dtype=dtype),
      'grad': np.array([[0.5, -0.2], [-0.1, 0.4]], dtype=dtype),
      'indices': np.array([0, 1], dtype=np.int32),
      'use_locking': False,
      'name': "s_a_adadelta_neg_var"
  }))

  # Input 10: Empty indices
  dtype = np.float32
  list_of_inputs.append(copy.deepcopy({
      'var': np.random.rand(10, 5).astype(dtype),
      'accum': np.random.rand(10, 5).astype(dtype),
      'accum_update': np.random.rand(10, 5).astype(dtype),
      'lr': np.array(0.01, dtype=dtype),
      'rho': np.array(0.9, dtype=dtype),
      'epsilon': np.array(1e-8, dtype=dtype),
      'grad': np.empty((0, 5), dtype=dtype),
      'indices': np.array([], dtype=np.int32),
      'use_locking': False,
      'name': "s_a_adadelta_empty_indices"
  }))

  return list_of_inputs

generated_inputs["tf.raw_ops.SparseApplyAdadelta"] = tf_raw_ops_SparseApplyAdadelta_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseApplyAdadelta' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyAdadelta'.")

check_valid('tf.raw_ops.SparseApplyAdadelta', generated_inputs['tf.raw_ops.SparseApplyAdadelta'], lib="tf", suffix=0)
