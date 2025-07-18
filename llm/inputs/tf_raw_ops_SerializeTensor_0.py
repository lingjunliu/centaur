
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_serialize_tensor_inputs():
  """
  Generates a list of valid inputs for tf.raw_ops.SerializeTensor.
  """
  list_of_inputs = []

  # Input 1: Simple 1D integer tensor
  tensor = np.array([1, 2, 3, 4], dtype=np.int32)
  input_dict = {'tensor': tensor, 'name': 'serialize_int_tensor'}
  list_of_inputs.append(copy.deepcopy(input_dict))

  # Input 2: 2D float tensor with negative values
  tensor = np.array([[-1.1, 2.2], [3.3, -4.4]], dtype=np.float32)
  input_dict = {'tensor': tensor, 'name': 'serialize_float_tensor'}
  list_of_inputs.append(copy.deepcopy(input_dict))

  # Input 3: 3D boolean tensor
  tensor = np.array([[[True, False], [False, True]], [[False, False], [True, True]]], dtype=np.bool_)
  input_dict = {'tensor': tensor, 'name': 'serialize_bool_tensor'}
  list_of_inputs.append(copy.deepcopy(input_dict))

  # Input 4: Scalar (0D) tensor
  tensor = np.array(42, dtype=np.int64)
  input_dict = {'tensor': tensor, 'name': 'serialize_scalar'}
  list_of_inputs.append(copy.deepcopy(input_dict))

  # Input 5: Empty tensor
  tensor = np.array([], dtype=np.float32)
  input_dict = {'tensor': tensor, 'name': 'serialize_empty_tensor'}
  list_of_inputs.append(copy.deepcopy(input_dict))

  # Input 6: 4D uint8 tensor
  tensor = np.arange(2 * 3 * 4 * 5, dtype=np.uint8).reshape(2, 3, 4, 5)
  input_dict = {'tensor': tensor, 'name': 'serialize_4d_uint8'}
  list_of_inputs.append(copy.deepcopy(input_dict))

  # Input 7: Complex number tensor
  tensor = np.array([1+2j, 3+4j, 5-6j], dtype=np.complex64)
  input_dict = {'tensor': tensor, 'name': 'serialize_complex_tensor'}
  list_of_inputs.append(copy.deepcopy(input_dict))

  # Input 8: String tensor (using object dtype for bytes)
  tensor = np.array([b'hello', b'world', b'tensorflow'], dtype=np.object_)
  input_dict = {'tensor': tensor, 'name': 'serialize_string_tensor'}
  list_of_inputs.append(copy.deepcopy(input_dict))

  # Input 9: Large tensor with double precision
  tensor = np.zeros((10, 10, 10), dtype=np.float64)
  input_dict = {'tensor': tensor, 'name': 'serialize_large_tensor'}
  list_of_inputs.append(copy.deepcopy(input_dict))

  # Input 10: 2D tensor with name=None
  tensor = np.array([[10, 20], [30, 40]], dtype=np.int16)
  input_dict = {'tensor': tensor, 'name': None}
  list_of_inputs.append(copy.deepcopy(input_dict))

  # Input 11: A tensor with mixed signs and zero
  tensor = np.array([-10, 0, 10, -5.5, 5.5, 0.0], dtype=np.float32)
  input_dict = {'tensor': tensor, 'name': 'serialize_mixed_sign'}
  list_of_inputs.append(copy.deepcopy(input_dict))

  # Input 12: Tensor with a zero-sized dimension
  tensor = np.zeros((5, 0, 5), dtype=np.int32)
  input_dict = {'tensor': tensor, 'name': 'serialize_zero_dim_tensor'}
  list_of_inputs.append(copy.deepcopy(input_dict))

  return list_of_inputs

generated_inputs["tf.raw_ops.SerializeTensor"] = tf_raw_ops_serialize_tensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SerializeTensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SerializeTensor'.")

check_valid('tf.raw_ops.SerializeTensor', generated_inputs['tf.raw_ops.SerializeTensor'], lib="tf", suffix=0)
