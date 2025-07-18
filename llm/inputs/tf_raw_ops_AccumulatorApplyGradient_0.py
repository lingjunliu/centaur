
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_accumulatorapplygradient_inputs():
  """
  Generates a list of valid inputs for tf.raw_ops.AccumulatorApplyGradient.
  NOTE: This operation is not supported in Eager execution mode, which is the
  default in modern TensorFlow. Executing this op in an eager context will
  raise a RuntimeError. The generated inputs are syntactically valid according
  to the API signature but are expected to fail at runtime in such an environment.
  """
  list_of_inputs = []

  # Input 1: Basic float32 gradient, 1D
  input_dict_1 = {
      'handle': np.array("acc_handle_1", dtype=object),
      'local_step': np.array(100, dtype=np.int64),
      'gradient': np.array([1.0, 2.5, -3.0], dtype=np.float32),
      'name': 'apply_grad_float32'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_1))

  # Input 2: int32 gradient, 2D, no name
  input_dict_2 = {
      'handle': np.array("acc_handle_2", dtype=object),
      'local_step': np.array(200, dtype=np.int64),
      'gradient': np.array([[1, 2], [3, 4]], dtype=np.int32),
      'name': None
  }
  list_of_inputs.append(copy.deepcopy(input_dict_2))

  # Input 3: float64 gradient, scalar
  input_dict_3 = {
      'handle': np.array("acc_handle_3", dtype=object),
      'local_step': np.array(0, dtype=np.int64),
      'gradient': np.array(3.14159, dtype=np.float64),
      'name': 'apply_scalar_grad'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_3))

  # Input 4: complex128 gradient, 2D
  input_dict_4 = {
      'handle': np.array("acc_handle_4", dtype=object),
      'local_step': np.array(50, dtype=np.int64),
      'gradient': np.array([[1+2j, 3-4j], [-5+6j, 7+8j]], dtype=np.complex128),
      'name': 'apply_complex128_grad'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_4))

  # Input 5: uint8 gradient, 3D
  input_dict_5 = {
      'handle': np.array("acc_handle_5", dtype=object),
      'local_step': np.array(1, dtype=np.int64),
      'gradient': np.array([[[10, 20], [30, 40]], [[50, 60], [70, 80]]], dtype=np.uint8),
      'name': None
  }
  list_of_inputs.append(copy.deepcopy(input_dict_5))

  # Input 6: int8 gradient, 1D with negative values
  input_dict_6 = {
      'handle': np.array("acc_handle_1", dtype=object),
      'local_step': np.array(101, dtype=np.int64),
      'gradient': np.array([-128, 0, 127], dtype=np.int8),
      'name': 'apply_int8_grad'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_6))

  # Input 7: half (float16) gradient
  input_dict_7 = {
      'handle': np.array("acc_handle_7", dtype=object),
      'local_step': np.array(999, dtype=np.int64),
      'gradient': np.array([[0.5, -0.5], [1.5, -1.5]], dtype=np.float16),
      'name': 'apply_half_grad'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_7))

  # Input 8: int64 gradient (matching local_step type)
  input_dict_8 = {
      'handle': np.array("acc_handle_8", dtype=object),
      'local_step': np.array(42, dtype=np.int64),
      'gradient': np.array([9223372036854775807, -9223372036854775808], dtype=np.int64),
      'name': None
  }
  list_of_inputs.append(copy.deepcopy(input_dict_8))

  # Input 9: complex64 gradient
  input_dict_9 = {
      'handle': np.array("acc_handle_9", dtype=object),
      'local_step': np.array(12345, dtype=np.int64),
      'gradient': np.array([1+1j, -1-1j], dtype=np.complex64),
      'name': 'apply_c64_grad'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_9))

  # Input 10: uint16 gradient
  input_dict_10 = {
      'handle': np.array("acc_handle_10", dtype=object),
      'local_step': np.array(5, dtype=np.int64),
      'gradient': np.array([0, 65535, 1000], dtype=np.uint16),
      'name': 'apply_uint16_grad'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_10))

  return list_of_inputs

generated_inputs["tf.raw_ops.AccumulatorApplyGradient"] = tf_raw_ops_accumulatorapplygradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.AccumulatorApplyGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AccumulatorApplyGradient'.")

check_valid('tf.raw_ops.AccumulatorApplyGradient', generated_inputs['tf.raw_ops.AccumulatorApplyGradient'], lib="tf", suffix=0)
