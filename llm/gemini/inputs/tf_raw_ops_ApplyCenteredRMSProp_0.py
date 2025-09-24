
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_apply_centered_rms_prop_inputs():
  """
  Generates a list of valid inputs for tf.raw_ops.ApplyCenteredRMSProp.
  This version provides all tensor inputs as numpy arrays, which is required
  by the user's test harness that causes an `AttributeError` with `tf.Variable`.
  Note: This API requires mutable tensor inputs (`tf.Variable`) for eager
  execution. Passing numpy arrays may lead to a `RuntimeError` during the
  actual API call if the test harness does not automatically convert them to
  `tf.Variable`.
  """
  list_of_inputs = []

  # Input 1: Basic float32, 1D
  input_dict_1 = {
      'var': np.array([1.0, 2.0, 3.0], dtype=np.float32),
      'mg': np.array([0.1, 0.2, 0.3], dtype=np.float32),
      'ms': np.array([0.01, 0.02, 0.03], dtype=np.float32),
      'mom': np.array([0.0, 0.0, 0.0], dtype=np.float32),
      'lr': np.array(0.001, dtype=np.float32),
      'rho': np.array(0.9, dtype=np.float32),
      'momentum': np.array(0.0, dtype=np.float32),
      'epsilon': np.array(1e-7, dtype=np.float32),
      'grad': np.array([0.5, -0.5, 0.2], dtype=np.float32),
      'use_locking': False,
      'name': 'test_float32_1d'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_1))

  # Input 2: float64, 2D, with locking and momentum
  input_dict_2 = {
      'var': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
      'mg': np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
      'ms': np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float64),
      'mom': np.array([[0.1, -0.1], [0.2, -0.2]], dtype=np.float64),
      'lr': np.array(0.01, dtype=np.float64),
      'rho': np.array(0.95, dtype=np.float64),
      'momentum': np.array(0.8, dtype=np.float64),
      'epsilon': np.array(1e-8, dtype=np.float64),
      'grad': np.random.randn(2, 2).astype(np.float64),
      'use_locking': True,
      'name': 'test_float64_2d'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_2))

  # Input 3: Zero gradient
  input_dict_3 = {
      'var': np.array([10.0, 20.0], dtype=np.float32),
      'mg': np.array([1.0, 2.0], dtype=np.float32),
      'ms': np.array([0.1, 0.2], dtype=np.float32),
      'mom': np.array([0.5, 0.5], dtype=np.float32),
      'lr': np.array(0.001, dtype=np.float32),
      'rho': np.array(0.9, dtype=np.float32),
      'momentum': np.array(0.5, dtype=np.float32),
      'epsilon': np.array(1e-7, dtype=np.float32),
      'grad': np.zeros((2,), dtype=np.float32),
      'use_locking': False,
      'name': 'test_zero_grad'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_3))

  # Input 4: half precision (float16)
  input_dict_4 = {
      'var': np.array([1.0, 2.0], dtype=np.half),
      'mg': np.array([0.1, 0.2], dtype=np.half),
      'ms': np.array([0.01, 0.02], dtype=np.half),
      'mom': np.array([0.0, 0.0], dtype=np.half),
      'lr': np.array(0.01, dtype=np.half),
      'rho': np.array(0.9, dtype=np.half),
      'momentum': np.array(0.1, dtype=np.half),
      'epsilon': np.array(1e-4, dtype=np.half),
      'grad': np.array([0.3, -0.4], dtype=np.half),
      'use_locking': False,
      'name': 'test_half_precision'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_4))

  # Input 5: Scalar variable
  input_dict_5 = {
      'var': np.array(100.0, dtype=np.float32),
      'mg': np.array(1.0, dtype=np.float32),
      'ms': np.array(0.5, dtype=np.float32),
      'mom': np.array(0.2, dtype=np.float32),
      'lr': np.array(0.1, dtype=np.float32),
      'rho': np.array(0.99, dtype=np.float32),
      'momentum': np.array(0.9, dtype=np.float32),
      'epsilon': np.array(1e-6, dtype=np.float32),
      'grad': np.array(-5.0, dtype=np.float32),
      'use_locking': False,
      'name': 'test_scalar'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_5))

  # Input 6: complex64
  input_dict_6 = {
      'var': np.array([1.0+2.0j, 3.0-1.0j], dtype=np.complex64),
      'mg': np.array([0.1+0.1j, 0.2-0.2j], dtype=np.complex64),
      'ms': np.array([0.5, 0.6], dtype=np.complex64), # `ms` is usually real-valued
      'mom': np.array([0.0+0.0j, 0.0+0.0j], dtype=np.complex64),
      'lr': np.array(0.01, dtype=np.complex64),
      'rho': np.array(0.9, dtype=np.complex64),
      'momentum': np.array(0.8, dtype=np.complex64),
      'epsilon': np.array(1e-7, dtype=np.complex64),
      'grad': np.array([0.3+0.4j, -0.1-0.2j], dtype=np.complex64),
      'use_locking': False,
      'name': 'test_complex64'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_6))

  # Input 7: High dimensional (3D)
  shape = (2, 2, 2)
  input_dict_7 = {
      'var': np.random.rand(*shape).astype(np.float32),
      'mg': np.random.rand(*shape).astype(np.float32),
      'ms': np.abs(np.random.rand(*shape).astype(np.float32)),
      'mom': np.random.rand(*shape).astype(np.float32),
      'lr': np.array(0.001, dtype=np.float32),
      'rho': np.array(0.9, dtype=np.float32),
      'momentum': np.array(0.5, dtype=np.float32),
      'epsilon': np.array(1e-7, dtype=np.float32),
      'grad': np.random.randn(*shape).astype(np.float32),
      'use_locking': False,
      'name': 'test_high_dim'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_7))

  # Input 8: complex128
  input_dict_8 = {
      'var': np.array([1.0+2.0j], dtype=np.complex128),
      'mg': np.array([0.1+0.1j], dtype=np.complex128),
      'ms': np.array([0.5], dtype=np.complex128),
      'mom': np.array([0.0+0.0j], dtype=np.complex128),
      'lr': np.array(0.01, dtype=np.complex128),
      'rho': np.array(0.9, dtype=np.complex128),
      'momentum': np.array(0.8, dtype=np.complex128),
      'epsilon': np.array(1e-7, dtype=np.complex128),
      'grad': np.array([0.3+0.4j], dtype=np.complex128),
      'use_locking': False,
      'name': 'test_complex128'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_8))
  
  return list_of_inputs

generated_inputs["tf.raw_ops.ApplyCenteredRMSProp"] = tf_raw_ops_apply_centered_rms_prop_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyCenteredRMSProp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyCenteredRMSProp'.")

check_valid('tf.raw_ops.ApplyCenteredRMSProp', generated_inputs['tf.raw_ops.ApplyCenteredRMSProp'], lib="tf", suffix=0)
