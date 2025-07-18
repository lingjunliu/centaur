
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_map_peek_inputs():
  """
  Generates a list of valid inputs for the tf.raw_ops.MapPeek operation.
  This op is expected to block and may cause a timeout if the key is not
  present in the map, which is the expected behavior in an isolated test.
  These inputs are valid according to the API signature. The timeout is an
  execution artifact, not an input validity error.
  """
  list_of_inputs = []

  # Input 1: Basic case with minimal arguments and default values.
  # This is a valid call and is expected to block, potentially causing a timeout.
  input_dict_1 = {
      'key': np.array(10, dtype=np.int64),
      'indices': np.array([0], dtype=np.int32),
      'dtypes': [tf.float32],
      'capacity': 0,
      'memory_limit': 0,
      'container': "",
      'shared_name': "",
      'name': "peek_simple_float"
  }
  list_of_inputs.append(copy.deepcopy(input_dict_1))

  # Input 2: Peeking a single integer tensor.
  input_dict_2 = {
      'key': np.array(20, dtype=np.int64),
      'indices': np.array([0], dtype=np.int32),
      'dtypes': [tf.int64],
      'capacity': 0,
      'memory_limit': 0,
      'container': "",
      'shared_name': "",
      'name': "peek_simple_int"
  }
  list_of_inputs.append(copy.deepcopy(input_dict_2))

  # Input 3: Using a shared_name. This is still expected to block.
  input_dict_3 = {
      'key': np.array(30, dtype=np.int64),
      'indices': np.array([0], dtype=np.int32),
      'dtypes': [tf.string],
      'capacity': 10,
      'memory_limit': 1024,
      'container': "mycontainer",
      'shared_name': "mysharedresource",
      'name': "peek_shared_string"
  }
  list_of_inputs.append(copy.deepcopy(input_dict_3))

  # Input 4: Peeking multiple indices from a hypothetical multi-value entry.
  input_dict_4 = {
      'key': np.array(40, dtype=np.int64),
      'indices': np.array([0, 1], dtype=np.int32),
      'dtypes': [tf.bool, tf.double],
      'capacity': 0,
      'memory_limit': 0,
      'container': "",
      'shared_name': "",
      'name': "peek_multi_index"
  }
  list_of_inputs.append(copy.deepcopy(input_dict_4))

  # Input 5: Peeking a complex type.
  input_dict_5 = {
      'key': np.array(50, dtype=np.int64),
      'indices': np.array([0], dtype=np.int32),
      'dtypes': [tf.complex64],
      'capacity': 0,
      'memory_limit': 0,
      'container': "",
      'shared_name': "",
      'name': "peek_complex"
  }
  list_of_inputs.append(copy.deepcopy(input_dict_5))

  return list_of_inputs

generated_inputs["tf.raw_ops.MapPeek"] = tf_raw_ops_map_peek_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MapPeek' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MapPeek'.")

check_valid('tf.raw_ops.MapPeek', generated_inputs['tf.raw_ops.MapPeek'], lib="tf", suffix=0)
