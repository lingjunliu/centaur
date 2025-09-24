
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_refselect_inputs():
  """
  Generates a list of inputs for tf.raw_ops.RefSelect.
  """
  list_of_inputs = []

  # Input 1: Select the first of two 1D float tensors.
  input_dict_1 = {
      'index': np.array(0, dtype=np.int32),
      'inputs': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
      'name': 'select_first_float'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_1))

  # Input 2: Select the last of three 2D int tensors.
  input_dict_2 = {
      'index': np.array(2, dtype=np.int32),
      'inputs': np.array([[[1, 2]], [[3, 4]], [[5, 6]]], dtype=np.int32),
      'name': 'select_last_int'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_2))

  # Input 3: Select from a list of scalars (0D tensors).
  input_dict_3 = {
      'index': np.array(1, dtype=np.int32),
      'inputs': np.array([100, 200, 300], dtype=np.int64),
      'name': 'select_from_scalars'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_3))

  # Input 4: Select from a list of complex tensors.
  input_dict_4 = {
      'index': np.array(0, dtype=np.int32),
      'inputs': np.array([[1+2j], [3+4j]], dtype=np.complex64),
      'name': 'select_complex'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_4))

  # Input 5: Select from a list of boolean tensors.
  input_dict_5 = {
      'index': np.array(1, dtype=np.int32),
      'inputs': np.array([[True], [False]], dtype=np.bool_),
      'name': 'select_bool'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_5))

  # Input 6: Using uint8 tensors.
  input_dict_6 = {
      'index': np.array(3, dtype=np.int32),
      'inputs': np.arange(12, dtype=np.uint8).reshape(4, 3),
      'name': 'select_uint8'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_6))
  
  # Input 7: Using 3D tensors.
  input_dict_7 = {
      'index': np.array(1, dtype=np.int32),
      'inputs': np.stack([
          np.ones((2, 2, 2), dtype=np.float32),
          np.zeros((2, 2, 2), dtype=np.float32)
      ]),
      'name': 'select_3d'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_7))

  # Input 8: name is None
  input_dict_8 = {
      'index': np.array(0, dtype=np.int32),
      'inputs': np.array([[-5.0, -10.0], [5.0, 10.0]], dtype=np.float64),
      'name': None
  }
  list_of_inputs.append(copy.deepcopy(input_dict_8))

  # Input 9: Large number of inputs to select from.
  input_dict_9 = {
      'index': np.array(19, dtype=np.int32),
      'inputs': np.arange(20, dtype=np.int16).reshape(20, 1),
      'name': 'select_from_many'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_9))

  # Input 10: Select from complex128 tensors.
  input_dict_10 = {
      'index': np.array(2, dtype=np.int32),
      'inputs': np.array([[1+1j],[2+2j],[3+3j]], dtype=np.complex128),
      'name': 'select_complex128'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_10))

  return list_of_inputs

generated_inputs["tf.raw_ops.RefSelect"] = get_tf_raw_ops_refselect_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RefSelect' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RefSelect'.")

check_valid('tf.raw_ops.RefSelect', generated_inputs['tf.raw_ops.RefSelect'], lib="tf", suffix=0)
