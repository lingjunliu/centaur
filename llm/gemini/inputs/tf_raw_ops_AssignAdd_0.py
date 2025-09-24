
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_assign_add_inputs():
  """
  Generates a list of valid inputs for the tf.raw_ops.AssignAdd operation.
  Note: This raw op is designed for graph-mode and will raise a RuntimeError
  in eager execution. The inputs provided are valid for the op's signature
  and would work in a graph context (e.g., inside a @tf.function).
  """
  list_of_inputs = []
  bfloat16_dtype = tf.bfloat16.as_numpy_dtype

  # The execution harness requires a `.size` attribute, which we add to the variable.
  def create_variable_with_size(np_array):
    var = tf.Variable(np_array)
    var.size = np_array.size
    return var

  # Input 1: Basic float32 addition
  input_1 = {
      'ref': create_variable_with_size(np.array([1.0, 2.0, 3.0], dtype=np.float32)),
      'value': np.array([0.5, 0.5, 0.5], dtype=np.float32),
      'use_locking': False,
      'name': 'add_float32'
  }
  list_of_inputs.append(input_1)

  # Input 2: 2D int32 with negative values and locking
  input_2 = {
      'ref': create_variable_with_size(np.array([[10, -20], [30, -40]], dtype=np.int32)),
      'value': np.array([[-5, 25], [-15, 45]], dtype=np.int32),
      'use_locking': True,
      'name': 'add_int32_2d_locked'
  }
  list_of_inputs.append(input_2)

  # Input 3: Scalar int64
  input_3 = {
      'ref': create_variable_with_size(np.array(100, dtype=np.int64)),
      'value': np.array(50, dtype=np.int64),
      'use_locking': False,
      'name': 'add_int64_scalar'
  }
  list_of_inputs.append(input_3)

  # Input 4: float64 tensors
  input_4 = {
      'ref': create_variable_with_size(np.array([1.23e4, 5.67e-2], dtype=np.float64)),
      'value': np.array([-1.0e4, 4.33e-2], dtype=np.float64),
      'use_locking': False,
      'name': 'add_float64'
  }
  list_of_inputs.append(input_4)

  # Input 5: uint8 with no optional name
  input_5 = {
      'ref': create_variable_with_size(np.array([[0, 10], [250, 100]], dtype=np.uint8)),
      'value': np.array([[5, 10], [5, 100]], dtype=np.uint8),
      'use_locking': True,
      'name': None
  }
  list_of_inputs.append(input_5)

  # Input 6: complex64 addition
  input_6 = {
      'ref': create_variable_with_size(np.array([1+2j, 3+4j], dtype=np.complex64)),
      'value': np.array([5-1j, -2+0j], dtype=np.complex64),
      'use_locking': False,
      'name': 'add_complex64'
  }
  list_of_inputs.append(input_6)

  # Input 7: bfloat16 addition
  input_7 = {
      'ref': create_variable_with_size(np.array([1.0, 2.0], dtype=bfloat16_dtype)),
      'value': np.array([0.125, -0.25], dtype=bfloat16_dtype),
      'use_locking': True,
      'name': 'add_bfloat16'
  }
  list_of_inputs.append(input_7)

  # Input 8: half (float16) addition
  input_8 = {
      'ref': create_variable_with_size(np.array([1.5, -2.5, 0.0], dtype=np.float16)),
      'value': np.array([0.5, 0.5, 1.0], dtype=np.float16),
      'use_locking': False,
      'name': 'add_half'
  }
  list_of_inputs.append(input_8)

  # Input 9: 3D int16, adding zero
  input_9 = {
      'ref': create_variable_with_size(np.array([[[100], [200]], [[-300], [400]]], dtype=np.int16)),
      'value': np.array([[[0], [0]], [[0], [0]]], dtype=np.int16),
      'use_locking': False,
      'name': 'add_zero_int16'
  }
  list_of_inputs.append(input_9)

  # Input 10: large uint32
  input_10 = {
      'ref': create_variable_with_size(np.array([2**32 - 100], dtype=np.uint32)),
      'value': np.array([50], dtype=np.uint32),
      'use_locking': True,
      'name': 'add_large_uint32'
  }
  list_of_inputs.append(input_10)

  return list_of_inputs

generated_inputs["tf.raw_ops.AssignAdd"] = tf_raw_ops_assign_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.AssignAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AssignAdd'.")

check_valid('tf.raw_ops.AssignAdd', generated_inputs['tf.raw_ops.AssignAdd'], lib="tf", suffix=0)
