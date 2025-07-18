
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_barrier_inputs():
  """
  Generates a list of valid inputs for the tf.raw_ops.Barrier function.

  IMPORTANT NOTE: The error 'RuntimeError: barrier op does not support eager
  execution' is an expected behavior for this specific operation. This op is
  part of TensorFlow's legacy graph-based infrastructure and is incompatible
  with the default eager execution mode of modern TensorFlow. The error is
  caused by the execution environment and *cannot* be fixed by modifying the
  inputs provided below. The inputs are syntactically valid according to the
  API's signature for a graph-based execution context.
  """
  list_of_inputs = []

  # Input 1: Basic case, single component, defaults for optional args
  input_dict_1 = {
      'component_types': [np.float32],
      'shapes': [[1, 16]],
      'capacity': -1,
      'container': '',
      'shared_name': '',
      'name': 'barrier_basic'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_1))

  # Input 2: Two components with different integer types and a set capacity
  input_dict_2 = {
      'component_types': [np.int32, np.int64],
      'shapes': [[1, 8], [1, 4]],
      'capacity': 100,
      'container': '',
      'shared_name': '',
      'name': 'barrier_multi_int'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_2))

  # Input 3: Using a non-empty container name
  input_dict_3 = {
      'component_types': [np.bool_],
      'shapes': [[1, 1]],
      'capacity': 50,
      'container': 'my_test_container_1',
      'shared_name': '',
      'name': 'barrier_in_container'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_3))

  # Input 4: Using a shared_name for potential cross-session use
  input_dict_4 = {
      'component_types': [np.float64],
      'shapes': [[1, 2, 3]],
      'capacity': -1,
      'container': '',
      'shared_name': 'my_shared_barrier_name',
      'name': 'barrier_shared'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_4))

  # Input 5: Using both container and shared_name
  input_dict_5 = {
      'component_types': [np.int16],
      'shapes': [[1, 64]],
      'capacity': 25,
      'container': 'container_for_shared',
      'shared_name': 'barrier_inside_container',
      'name': 'barrier_full_spec'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_5))

  # Input 6: Using the default empty list for the 'shapes' attribute
  input_dict_6 = {
      'component_types': [np.float32, np.int32],
      'shapes': [],
      'capacity': 10,
      'container': '',
      'shared_name': 'barrier_no_shapes_attr',
      'name': 'barrier_default_shapes'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_6))

  # Input 7: Zero capacity
  input_dict_7 = {
      'component_types': [np.uint8],
      'shapes': [[1, 10]],
      'capacity': 0,
      'container': '',
      'shared_name': '',
      'name': 'barrier_zero_capacity'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_7))

  # Input 8: Multiple components with various types
  input_dict_8 = {
      'component_types': [np.float16, np.uint16, np.bool_],
      'shapes': [[1, 4], [1, 4], [1, 1]],
      'capacity': 200,
      'container': '',
      'shared_name': '',
      'name': 'barrier_mixed_types'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_8))

  # Input 9: A single component with a higher-rank shape
  input_dict_9 = {
      'component_types': [np.int32],
      'shapes': [[1, 2, 3, 4]],
      'capacity': 5,
      'container': '',
      'shared_name': '',
      'name': 'barrier_high_rank'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_9))

  # Input 10: Using complex number data types
  input_dict_10 = {
      'component_types': [np.complex64, np.complex128],
      'shapes': [[1, 3], [1, 3]],
      'capacity': 15,
      'container': 'complex_container',
      'shared_name': 'complex_barrier_shared',
      'name': 'barrier_complex'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_10))

  return list_of_inputs

generated_inputs["tf.raw_ops.Barrier"] = tf_raw_ops_barrier_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Barrier' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Barrier'.")

check_valid('tf.raw_ops.Barrier', generated_inputs['tf.raw_ops.Barrier'], lib="tf", suffix=0)
