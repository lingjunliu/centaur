
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_refnextiteration_inputs():
  """
  Generates a list of valid inputs for tf.raw_ops.RefNextIteration.
  """
  list_of_inputs = []

  # Input 1: 1D float32 tensor
  list_of_inputs.append(
      {
          'data': np.array([1.0, 2.0, 3.0], dtype=np.float32),
          'name': 'float32_1d_v1'
      }
  )

  # Input 2: 2D int32 tensor
  list_of_inputs.append(
      {
          'data': np.array([[1, 2], [3, 4]], dtype=np.int32),
          'name': 'int32_2d_v1'
      }
  )

  # Input 3: 3D uint8 tensor
  list_of_inputs.append(
      {
          'data': np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.uint8),
          'name': 'uint8_3d_v1'
      }
  )

  # Input 4: Scalar int64 tensor
  list_of_inputs.append(
      {
          'data': np.array(-500, dtype=np.int64),
          'name': 'int64_scalar_v1'
      }
  )

  # Input 5: 1D bool tensor
  list_of_inputs.append(
      {
          'data': np.array([True, False, False, True], dtype=np.bool_),
          'name': 'bool_1d_v1'
      }
  )

  # Input 6: 2D float64 tensor
  list_of_inputs.append(
      {
          'data': np.random.rand(3, 4).astype(np.float64),
          'name': 'float64_2d_v1'
      }
  )

  # Input 7: 4D int16 tensor
  list_of_inputs.append(
      {
          'data': np.ones((1, 2, 3, 1), dtype=np.int16),
          'name': 'int16_4d_v1'
      }
  )

  # Input 8: 2D complex64 tensor
  list_of_inputs.append(
      {
          'data': np.array([[1+1j, 2-2j], [3+3j, 4-4j]], dtype=np.complex64),
          'name': 'complex64_2d_v1'
      }
  )

  # Input 9: 1D tensor with negative float values
  list_of_inputs.append(
      {
          'data': np.array([-10.5, -20.0, -5.25], dtype=np.float32),
          'name': 'negative_float_1d_v1'
      }
  )

  # Input 10: 2D tensor of all zeros
  list_of_inputs.append(
      {
          'data': np.zeros((5, 5), dtype=np.int32),
          'name': 'zeros_2d_v1'
      }
  )

  return [copy.deepcopy(i) for i in list_of_inputs]

generated_inputs["tf.raw_ops.RefNextIteration"] = get_refnextiteration_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RefNextIteration' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RefNextIteration'.")

check_valid('tf.raw_ops.RefNextIteration', generated_inputs['tf.raw_ops.RefNextIteration'], lib="tf", suffix=0)
