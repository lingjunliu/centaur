
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.compat.v1.disable_eager_execution()

def tf_raw_ops_conditional_accumulator_inputs():
  """
  Generates a list of valid inputs for the tf.raw_ops.ConditionalAccumulator function.
  """
  list_of_inputs = []

  # Input 1: Basic case with float32 and default options.
  input_dict_1 = {
      'dtype': np.float32,
      'shape': [10],
      'container': '',
      'shared_name': '',
      'reduction_type': 'MEAN',
      'name': 'acc_float32'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_1))

  # Input 2: Integer type (int32) with a 2D shape and 'SUM' reduction.
  input_dict_2 = {
      'dtype': np.int32,
      'shape': [5, 5],
      'container': '',
      'shared_name': '',
      'reduction_type': 'SUM',
      'name': 'acc_sum_int32'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_2))

  # Input 3: Scalar shape and float64 data type.
  input_dict_3 = {
      'dtype': np.float64,
      'shape': [],
      'container': '',
      'shared_name': '',
      'reduction_type': 'MEAN',
      'name': 'acc_scalar_float64'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_3))

  # Input 4: Using a non-empty container.
  input_dict_4 = {
      'dtype': np.half,
      'shape': [100],
      'container': 'my_container',
      'shared_name': '',
      'reduction_type': 'MEAN',
      'name': 'acc_container'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_4))

  # Input 5: Using a non-empty shared_name for cross-session sharing.
  input_dict_5 = {
      'dtype': np.int64,
      'shape': [2, 8],
      'container': '',
      'shared_name': 'my_shared_accumulator',
      'reduction_type': 'SUM',
      'name': 'acc_shared'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_5))

  # Input 6: Using both a non-empty container and shared_name.
  input_dict_6 = {
      'dtype': np.uint8,
      'shape': [3, 3, 3],
      'container': 'another_container',
      'shared_name': 'another_shared_name',
      'reduction_type': 'MEAN',
      'name': 'acc_full_spec'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_6))

  # Input 7: Complex number data type (complex64).
  input_dict_7 = {
      'dtype': np.complex64,
      'shape': [4, 4],
      'container': '',
      'shared_name': '',
      'reduction_type': 'MEAN',
      'name': 'acc_complex64'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_7))

  # Input 8: Signed 8-bit integer with a large 1D shape.
  input_dict_8 = {
      'dtype': np.int8,
      'shape': [2048],
      'container': '',
      'shared_name': '',
      'reduction_type': 'SUM',
      'name': 'acc_int8_large'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_8))

  # Input 9: Unsigned 16-bit integer with a 4D shape.
  input_dict_9 = {
      'dtype': np.uint16,
      'shape': [2, 2, 3, 4],
      'container': '',
      'shared_name': '',
      'reduction_type': 'MEAN',
      'name': 'acc_uint16_4d'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_9))

  # Input 10: Complex number data type (complex128).
  input_dict_10 = {
      'dtype': np.complex128,
      'shape': [2, 2],
      'container': 'c128_container',
      'shared_name': '',
      'reduction_type': 'SUM',
      'name': 'acc_complex128'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_10))

  return list_of_inputs

generated_inputs["tf.raw_ops.ConditionalAccumulator"] = tf_raw_ops_conditional_accumulator_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ConditionalAccumulator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ConditionalAccumulator'.")

check_valid('tf.raw_ops.ConditionalAccumulator', generated_inputs['tf.raw_ops.ConditionalAccumulator'], lib="tf", suffix=0)
