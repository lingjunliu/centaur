
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_assign_add_inputs():
  """
  Generates a list of valid inputs for tf.raw_ops.AssignAdd.
  The 'ref' input must be a tf.Variable to be mutable, as required by the
  operation in eager execution mode to prevent a RuntimeError.
  """
  list_of_inputs = []

  # Input 1: Basic float32, 1D
  input_dict_1 = {
      'ref': tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32)),
      'value': np.array([0.5, 0.5, 0.5], dtype=np.float32),
      'use_locking': False,
      'name': 'add_floats'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_1))

  # Input 2: 2D int32 with negative values and locking
  input_dict_2 = {
      'ref': tf.Variable(np.array([[-1, 2], [-3, 4]], dtype=np.int32)),
      'value': np.array([[1, -1], [3, -3]], dtype=np.int32),
      'use_locking': True,
      'name': 'add_ints_2d'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_2))

  # Input 3: Scalar float64, no name
  input_dict_3 = {
      'ref': tf.Variable(100.0, dtype=tf.float64),
      'value': np.array(-50.5, dtype=np.float64),
      'use_locking': False,
      'name': None
  }
  list_of_inputs.append(copy.deepcopy(input_dict_3))

  # Input 4: 3D int64 with locking
  input_dict_4 = {
      'ref': tf.Variable(np.arange(8, dtype=np.int64).reshape((2, 2, 2))),
      'value': np.ones((2, 2, 2), dtype=np.int64),
      'use_locking': True,
      'name': 'add_int64_3d'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_4))

  # Input 5: 1D complex64
  input_dict_5 = {
      'ref': tf.Variable(np.array([1+2j, 3+4j], dtype=np.complex64)),
      'value': np.array([5-1j, -2+0j], dtype=np.complex64),
      'use_locking': False,
      'name': 'add_complex64'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_5))

  # Input 6: 2D complex128 with locking, no name
  input_dict_6 = {
      'ref': tf.Variable(np.array([[1.5+2.5j]], dtype=np.complex128)),
      'value': np.array([[-0.5-1.5j]], dtype=np.complex128),
      'use_locking': True,
      'name': None
  }
  list_of_inputs.append(copy.deepcopy(input_dict_6))
  
  # Input 7: 1D uint8
  input_dict_7 = {
      'ref': tf.Variable(np.array([10, 20, 250], dtype=np.uint8)),
      'value': np.array([5, 5, 5], dtype=np.uint8),
      'use_locking': False,
      'name': 'add_uint8'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_7))

  # Input 8: 1D half (float16) with locking
  input_dict_8 = {
      'ref': tf.Variable(np.array([1.0, -1.0, 0.0], dtype=np.float16)),
      'value': np.array([0.1, 0.2, 0.3], dtype=np.float16),
      'use_locking': True,
      'name': 'add_half'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_8))

  # Input 9: 1D int8, no name
  input_dict_9 = {
      'ref': tf.Variable(np.array([120, -120], dtype=np.int8)),
      'value': np.array([5, 5], dtype=np.int8),
      'use_locking': False,
      'name': None
  }
  list_of_inputs.append(copy.deepcopy(input_dict_9))

  # Input 10: 2D uint16 with locking
  input_dict_10 = {
      'ref': tf.Variable(np.array([[1000], [2000]], dtype=np.uint16)),
      'value': np.array([[500], [500]], dtype=np.uint16),
      'use_locking': True,
      'name': 'add_uint16'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_10))

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
