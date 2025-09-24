
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_meshgrid_inputs():
  """
  Generates a list of valid inputs for the tf.meshgrid function.
  """
  list_of_inputs = []

  # Input 1: Basic 2D case from documentation example
  input_dict_1 = {
      'args': np.stack([np.array([1, 2, 3], dtype=np.int32), np.array([4, 5, 6], dtype=np.int32)]),
      'indexing': 'xy',
      'name': 'basic_xy'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_1))

  # Input 2: Basic 2D case with 'ij' indexing
  input_dict_2 = {
      'args': np.stack([np.array([1, 2, 3], dtype=np.int32), np.array([4, 5, 6], dtype=np.int32)]),
      'indexing': 'ij',
      'name': 'basic_ij'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_2))

  # Input 3: 3D grid with 'xy' indexing
  input_dict_3 = {
      'args': np.stack([np.array([0, 1], dtype=np.int32), np.array([2, 3], dtype=np.int32), np.array([4, 5], dtype=np.int32)]),
      'indexing': 'xy',
      'name': 'grid_3d_xy'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_3))

  # Input 4: 3D grid with 'ij' indexing and float type
  input_dict_4 = {
      'args': np.stack([np.array([0., 1.], dtype=np.float32), np.array([2., 3.], dtype=np.float32), np.array([4., 5.], dtype=np.float32)]),
      'indexing': 'ij',
      'name': 'grid_3d_ij_float'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_4))

  # Input 5: Inputs with negative values and float type
  input_dict_5 = {
      'args': np.stack([np.array([-1.5, -2.5], dtype=np.float32), np.array([0.5, -0.5], dtype=np.float32)]),
      'indexing': 'ij',
      'name': 'negative_floats'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_5))

  # Input 6: Single input array
  input_dict_6 = {
      'args': np.stack([np.array([100, 200, 300], dtype=np.int64)]),
      'indexing': 'xy',
      'name': 'single_arg'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_6))

  # Input 7: All empty input arrays
  input_dict_7 = {
      'args': np.stack([np.array([], dtype=np.float32), np.array([], dtype=np.float32)]),
      'indexing': 'xy',
      'name': 'all_empty_args'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_7))

  # Input 8: 4D grid
  input_dict_8 = {
      'args': np.stack([np.array([1]), np.array([2]), np.array([3]), np.array([4])]),
      'indexing': 'ij',
      'name': 'grid_4d'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_8))

  # Input 9: float64 type
  input_dict_9 = {
      'args': np.stack([np.array([1.1, 2.2], dtype=np.float64), np.array([3.3, 4.4], dtype=np.float64)]),
      'indexing': 'xy',
      'name': 'float64_type'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_9))

  # Input 10: Mixed int and float types (same length)
  input_dict_10 = {
      'args': np.stack([np.array([1, 2, 3], dtype=np.int32), np.array([4.0, 5.0, 6.0], dtype=np.float32)]),
      'indexing': 'ij',
      'name': 'mixed_int_float'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_10))

  # Input 11: 3D grid with negative integers
  input_dict_11 = {
      'args': np.stack([np.array([-1, -2], dtype=np.int16), np.array([-3, -4], dtype=np.int16), np.array([-5, -6], dtype=np.int16)]),
      'indexing': 'xy',
      'name': 'grid_3d_negative_int'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_11))

  return list_of_inputs

generated_inputs["tf.meshgrid"] = get_tf_meshgrid_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.meshgrid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.meshgrid'.")

check_valid('tf.meshgrid', generated_inputs['tf.meshgrid'], lib="tf", suffix=0)
