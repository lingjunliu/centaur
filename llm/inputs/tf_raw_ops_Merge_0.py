
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_merge_inputs():
  """
  Generates a list of valid inputs for tf.raw_ops.Merge.
  The 'inputs' parameter is a list of tensors, but to satisfy the calling
  framework which expects a .shape attribute, we stack them into a single numpy
  array. The framework is expected to unstack them before the API call.
  """
  list_of_inputs = []

  # Input 1: Basic case with float32 tensors
  list_of_inputs.append({
      'inputs': np.stack([
          np.array([1.0, 2.0, 3.0], dtype=np.float32),
          np.array([4.0, 5.0, 6.0], dtype=np.float32)
      ]),
      'name': 'merge_float32'
  })

  # Input 2: 2D int32 tensors
  list_of_inputs.append({
      'inputs': np.stack([
          np.array([[1, 2], [3, 4]], dtype=np.int32),
          np.array([[5, 6], [7, 8]], dtype=np.int32)
      ]),
      'name': 'merge_int32_2d'
  })

  # Input 3: List with three int64 tensors
  list_of_inputs.append({
      'inputs': np.stack([
          np.array([100], dtype=np.int64),
          np.array([200], dtype=np.int64),
          np.array([300], dtype=np.int64)
      ]),
      'name': 'merge_three_inputs'
  })

  # Input 4: Scalar (0-D) tensors with negative values
  list_of_inputs.append({
      'inputs': np.stack([
          np.array(-10, dtype=np.int32),
          np.array(20, dtype=np.int32)
      ]),
      'name': 'merge_scalars'
  })

  # Input 5: Single tensor in the input list
  list_of_inputs.append({
      'inputs': np.stack([
          np.array([-1, -2, -3, -4], dtype=np.int16)
      ]),
      'name': 'merge_single_input'
  })

  # Input 6: Boolean tensors
  list_of_inputs.append({
      'inputs': np.stack([
          np.array([[True, False], [False, True]], dtype=np.bool_),
          np.array([[False, False], [True, True]], dtype=np.bool_)
      ]),
      'name': 'merge_bool'
  })

  # Input 7: Higher-dimensional (3D) float64 tensors
  list_of_inputs.append({
      'inputs': np.stack([
          np.arange(8, dtype=np.float64).reshape((2, 2, 2)),
          np.arange(8, 16, dtype=np.float64).reshape((2, 2, 2))
      ]),
      'name': 'merge_3d_float64'
  })

  # Input 8: Unsigned integer type uint8
  list_of_inputs.append({
      'inputs': np.stack([
          np.array([0, 128, 255], dtype=np.uint8),
          np.array([1, 127, 254], dtype=np.uint8)
      ]),
      'name': 'merge_uint8'
  })

  # Input 9: Complex numbers (complex64)
  list_of_inputs.append({
      'inputs': np.stack([
          np.array([1+2j, 3+4j], dtype=np.complex64),
          np.array([5-6j, -7+8j], dtype=np.complex64)
      ]),
      'name': 'merge_complex64'
  })

  # Input 10: Empty tensors (shape=(0,)) -> Stacks to (2,0)
  list_of_inputs.append({
      'inputs': np.stack([
          np.array([], dtype=np.float32),
          np.array([], dtype=np.float32)
      ]),
      'name': 'merge_empty'
  })

  # Input 11: Tensors with a zero dimension -> Stacks to (2, 3, 0)
  list_of_inputs.append({
      'inputs': np.stack([
          np.zeros((3, 0), dtype=np.int32),
          np.zeros((3, 0), dtype=np.int32)
      ]),
      'name': 'merge_zero_dim'
  })

  # Input 12: A longer list of tensors
  list_of_inputs.append({
      'inputs': np.stack([
          np.array([1], dtype=np.int32),
          np.array([2], dtype=np.int32),
          np.array([3], dtype=np.int32),
          np.array([4], dtype=np.int32),
          np.array([5], dtype=np.int32)
      ]),
      'name': 'merge_long_list'
  })

  return list_of_inputs

generated_inputs["tf.raw_ops.Merge"] = tf_raw_ops_merge_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Merge' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Merge'.")

check_valid('tf.raw_ops.Merge', generated_inputs['tf.raw_ops.Merge'], lib="tf", suffix=0)
