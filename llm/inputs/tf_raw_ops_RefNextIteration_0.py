
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_refnextiteration_inputs():
  """
  Generates a list of syntactically valid inputs for tf.raw_ops.RefNextIteration.
  
  Note: The tf.raw_ops.RefNextIteration operation is designed exclusively
  for TensorFlow's graph mode and is not supported in eager execution. Calling
  this function in an eager context will always result in a RuntimeError.
  The inputs are provided to satisfy the generation requirement, even though
  they will fail in the provided testing environment.
  """
  list_of_inputs = []

  # Input 1: Basic 1D float32 tensor
  list_of_inputs.append({
      'data': np.array([1.1, 2.2, 3.3], dtype=np.float32),
      'name': 'iter_float_1d'
  })

  # Input 2: Basic 2D int32 tensor
  list_of_inputs.append({
      'data': np.array([[10, 20], [30, 40]], dtype=np.int32),
      'name': 'iter_int_2d'
  })

  # Input 3: Scalar float32
  list_of_inputs.append({
      'data': np.array(99.9, dtype=np.float32),
      'name': 'iter_scalar_float'
  })

  # Input 4: Scalar int32
  list_of_inputs.append({
      'data': np.array(-5, dtype=np.int32),
      'name': 'iter_scalar_int'
  })

  # Input 5: 3D float32 tensor
  list_of_inputs.append({
      'data': np.ones((2, 2, 2), dtype=np.float32),
      'name': 'iter_float_3d'
  })

  # Input 6: 1D int64 tensor
  list_of_inputs.append({
      'data': np.array([10000000000, -20000000000], dtype=np.int64),
      'name': 'iter_int64_1d'
  })

  # Input 7: 2D float64 tensor
  list_of_inputs.append({
      'data': np.random.randn(3, 2).astype(np.float64),
      'name': 'iter_float64_2d'
  })

  # Input 8: Empty tensor with a specific shape
  list_of_inputs.append({
      'data': np.empty((2, 0, 3), dtype=np.float32),
      'name': 'iter_empty'
  })

  # Input 9: A tensor containing only zero
  list_of_inputs.append({
      'data': np.array([0.0], dtype=np.float32),
      'name': 'iter_single_zero'
  })

  # Input 10: A larger tensor
  list_of_inputs.append({
      'data': np.linspace(0, 1, 10, dtype=np.float32).reshape(5, 2),
      'name': 'iter_linspace'
  })
  
  return [copy.deepcopy(d) for d in list_of_inputs]

generated_inputs["tf.raw_ops.RefNextIteration"] = tf_raw_ops_refnextiteration_inputs()

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
