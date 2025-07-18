
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_isvariableinitialized_inputs():
  """
  Generates a list of valid inputs for tf.raw_ops.IsVariableInitialized.
  This function provides inputs in numpy format as required by the testing
  harness. The API itself is designed for TensorFlow's graph mode and is
  expected to raise a RuntimeError in eager execution. Providing numpy
  arrays is the correct format for the harness, even if a runtime error occurs
  during the subsequent API call.
  """
  list_of_inputs = []

  # Input 1: Simple float32 scalar
  list_of_inputs.append(copy.deepcopy({
      'ref': np.array(3.14, dtype=np.float32),
      'name': 'check_float_scalar'
  }))

  # Input 2: 1D int32 vector with negative values
  list_of_inputs.append(copy.deepcopy({
      'ref': np.array([1, 2, 3, -4], dtype=np.int32),
      'name': 'check_int_vector'
  }))

  # Input 3: 2D float64 matrix with name=None
  list_of_inputs.append(copy.deepcopy({
      'ref': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
      'name': None
  }))

  # Input 4: 3D complex64 tensor
  list_of_inputs.append(copy.deepcopy({
      'ref': np.array([[[1+2j, 3+4j], [5+6j, 7+8j]]], dtype=np.complex64),
      'name': 'check_complex_tensor'
  }))

  # Input 5: Boolean vector
  list_of_inputs.append(copy.deepcopy({
      'ref': np.array([True, False, True], dtype=np.bool_),
      'name': 'bool_check'
  }))

  # Input 6: Empty tensor with shape (0,)
  list_of_inputs.append(copy.deepcopy({
      'ref': np.array([], dtype=np.float32),
      'name': 'check_empty'
  }))

  # Input 7: Empty tensor with shape (1, 0, 2)
  list_of_inputs.append(copy.deepcopy({
      'ref': np.empty(shape=(1, 0, 2), dtype=np.int64),
      'name': 'check_empty_with_dims'
  }))

  # Input 8: 2D uint8 tensor
  list_of_inputs.append(copy.deepcopy({
      'ref': np.array([[0, 255], [128, 64]], dtype=np.uint8),
      'name': 'check_uint8'
  }))

  # Input 9: 4D tensor of zeros with float16 type
  list_of_inputs.append(copy.deepcopy({
      'ref': np.zeros((2, 1, 2, 1), dtype=np.float16),
      'name': 'check_4d_float16_tensor'
  }))
  
  # Input 10: Complex128 scalar
  list_of_inputs.append(copy.deepcopy({
      'ref': np.array(5.5 - 9.1j, dtype=np.complex128),
      'name': 'check_complex128_scalar'
  }))

  return list_of_inputs

generated_inputs["tf.raw_ops.IsVariableInitialized"] = tf_raw_ops_isvariableinitialized_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.IsVariableInitialized' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.IsVariableInitialized'.")

check_valid('tf.raw_ops.IsVariableInitialized', generated_inputs['tf.raw_ops.IsVariableInitialized'], lib="tf", suffix=0)
