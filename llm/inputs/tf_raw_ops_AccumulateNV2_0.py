
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_accumulatenv2_inputs():
  """
  Generates a list of valid inputs for tf.raw_ops.AccumulateNV2.
  """
  list_of_inputs = []

  # Case 1: Basic float32 tensors, rank 1
  input_dict_1 = {
      'inputs': [np.array([1.1, 2.2, 3.3], dtype=np.float32), np.array([4.4, 5.5, 6.6], dtype=np.float32)],
      'shape': [3],
      'name': 'float32_sum_1d'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_1))

  # Case 2: 2D int32 tensors with negative values
  input_dict_2 = {
      'inputs': [np.array([[-1, 2], [-3, 4]], dtype=np.int32), np.array([[5, -6], [7, -8]], dtype=np.int32)],
      'shape': [2, 2],
      'name': 'int32_sum_2d'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_2))

  # Case 3: Scalar int64 tensors
  input_dict_3 = {
      'inputs': [np.array(10, dtype=np.int64), np.array(20, dtype=np.int64), np.array(-5, dtype=np.int64)],
      'shape': [],
      'name': 'scalar_int64_sum'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_3))

  # Case 4: 3D float64 tensors with no name
  input_dict_4 = {
      'inputs': [np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float64), np.array([[[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)],
      'shape': [1, 2, 2],
      'name': None
  }
  list_of_inputs.append(copy.deepcopy(input_dict_4))

  # Case 5: A larger list of uint8 tensors
  input_dict_5 = {
      'inputs': [
          np.array([1, 2, 3, 4], dtype=np.uint8),
          np.array([5, 6, 7, 8], dtype=np.uint8),
          np.array([9, 10, 11, 12], dtype=np.uint8),
          np.array([13, 14, 15, 16], dtype=np.uint8)
      ],
      'shape': [4],
      'name': 'uint8_sum_4_tensors'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_5))

  # Case 6: complex64 tensors
  input_dict_6 = {
      'inputs': [np.array([1+2j, 3+4j], dtype=np.complex64), np.array([5-1j, -2+3j], dtype=np.complex64)],
      'shape': [2],
      'name': 'complex64_sum'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_6))

  # Case 7: half precision (float16) tensors
  input_dict_7 = {
      'inputs': [
          np.array([[1.5, -2.5, 3.0], [4.0, 5.5, -6.0]], dtype=np.float16),
          np.array([[0.5, 0.5, 0.0], [-1.0, -1.5, 2.0]], dtype=np.float16)
      ],
      'shape': [2, 3],
      'name': 'half_precision_sum'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_7))

  # Case 8: int16 tensors
  input_dict_8 = {
      'inputs': [
          np.array([[100, -200, 300, -400, 500]], dtype=np.int16),
          np.array([[-50, 150, -250, 350, -450]], dtype=np.int16)
      ],
      'shape': [1, 5],
      'name': 'int16_sum'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_8))

  # Case 9: uint32 tensors
  input_dict_9 = {
      'inputs': [
          np.array([[[1, 2]], [[3, 4]]], dtype=np.uint32),
          np.array([[[10, 20]], [[30, 40]]], dtype=np.uint32),
          np.array([[[100, 200]], [[300, 400]]], dtype=np.uint32)
      ],
      'shape': [2, 1, 2],
      'name': 'uint32_sum'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_9))

  # Case 10: complex128 tensors
  input_dict_10 = {
      'inputs': [
          np.array([[1.23e10 + 4.56e10j]], dtype=np.complex128),
          np.array([[-7.89e10 - 1.23e10j]], dtype=np.complex128)
      ],
      'shape': [1, 1],
      'name': 'complex128_sum'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_10))

  return list_of_inputs

generated_inputs["tf.raw_ops.AccumulateNV2"] = tf_raw_ops_accumulatenv2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.AccumulateNV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AccumulateNV2'.")

check_valid('tf.raw_ops.AccumulateNV2', generated_inputs['tf.raw_ops.AccumulateNV2'], lib="tf", suffix=0)
