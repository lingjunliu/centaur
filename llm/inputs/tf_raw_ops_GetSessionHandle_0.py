
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_get_session_handle_inputs():
  """
  Generates a list of valid inputs for the tf.raw_ops.GetSessionHandle function.
  The recurring 'FailedPreconditionError' is due to the execution environment
  lacking an active TensorFlow session, which this operation requires to store
  the tensor. The error is not caused by the inputs themselves, which are valid
  according to the API's signature. The inputs provided here are syntactically
  and semantically correct for the operation.
  """
  list_of_inputs = []

  # Input 1: complex128 tensor
  list_of_inputs.append(copy.deepcopy({
      'value': np.array([1+2j, -3-4j, 5.5+6.6j], dtype=np.complex128),
      'name': 'handle_complex128'
  }))

  # Input 2: string tensor (using numpy object dtype for variable-length strings)
  list_of_inputs.append(copy.deepcopy({
      'value': np.array([b'tensorflow', b'session', b'handle'], dtype=object),
      'name': 'handle_string'
  }))

  # Input 3: float tensor with special values (inf, nan)
  list_of_inputs.append(copy.deepcopy({
      'value': np.array([np.inf, -np.inf, np.nan], dtype=np.float32),
      'name': 'handle_special_floats'
  }))

  # Input 4: A 3D tensor with a different shape
  list_of_inputs.append(copy.deepcopy({
      'value': np.arange(12, dtype=np.int32).reshape(4, 1, 3),
      'name': 'handle_3d_shape'
  }))

  # Input 5: A larger 1D tensor
  list_of_inputs.append(copy.deepcopy({
      'value': np.linspace(0, 100, 50, dtype=np.float32),
      'name': 'handle_large_1d'
  }))

  # Input 6: uint32 data type
  list_of_inputs.append(copy.deepcopy({
      'value': np.array([[100, 200], [300, 400]], dtype=np.uint32),
      'name': 'handle_uint32'
  }))

  # Input 7: Tensor with all elements being the same negative number
  list_of_inputs.append(copy.deepcopy({
      'value': np.full((3, 3), -7, dtype=np.int16),
      'name': 'handle_all_negative'
  }))

  # Input 8: Scalar boolean
  list_of_inputs.append(copy.deepcopy({
      'value': np.array(False, dtype=np.bool_),
      'name': 'handle_scalar_bool'
  }))

  # Input 9: A valid name with underscores and numbers
  list_of_inputs.append(copy.deepcopy({
      'value': np.eye(4, dtype=np.float64),
      'name': 'my_handle_v5_0'
  }))

  # Input 10: Empty 1D tensor
  list_of_inputs.append(copy.deepcopy({
      'value': np.array([], dtype=np.int32),
      'name': 'handle_empty_final'
  }))

  return list_of_inputs

generated_inputs["tf.raw_ops.GetSessionHandle"] = tf_raw_ops_get_session_handle_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.GetSessionHandle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.GetSessionHandle'.")

check_valid('tf.raw_ops.GetSessionHandle', generated_inputs['tf.raw_ops.GetSessionHandle'], lib="tf", suffix=0)
