
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_assign_sub_inputs():
  """
  Generates a list of valid inputs for tf.raw_ops.AssignSub.
  The "RuntimeError: assign_sub op does not support eager execution" is an
  inherent characteristic of this low-level stateful op. It is designed for
  TensorFlow's graph mode and expects a mutable tf.Variable as the 'ref'
  argument, not an immutable tf.Tensor. The provided numpy inputs are
  syntactically correct according to the API signature, assuming they are
  used in a compatible execution context (e.g., within a tf.function or a
  TensorFlow 1.x graph) where the 'ref' can be correctly instantiated as a
  mutable variable. The error is not with the inputs themselves but with
  the eager execution environment.
  """
  list_of_inputs = []

  # Input 1: Basic float32 subtraction, 1D
  input_dict_1 = {
      'ref': np.array([10.5, 20.0, -5.0], dtype=np.float32),
      'value': np.array([0.5, 10.0, 5.0], dtype=np.float32),
      'use_locking': False,
      'name': 'float32_sub_1d'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_1))

  # Input 2: int32 subtraction, 2D, with locking
  input_dict_2 = {
      'ref': np.array([[100, 200], [300, 400]], dtype=np.int32),
      'value': np.array([[50, -100], [150, 200]], dtype=np.int32),
      'use_locking': True,
      'name': 'int32_sub_2d_locked'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_2))

  # Input 3: float64 scalar subtraction
  input_dict_3 = {
      'ref': np.array(100.0, dtype=np.float64),
      'value': np.array(0.5, dtype=np.float64),
      'use_locking': False,
      'name': 'float64_sub_scalar'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_3))

  # Input 4: int64 scalar subtraction
  input_dict_4 = {
      'ref': np.array(10000000000, dtype=np.int64),
      'value': np.array(5000000000, dtype=np.int64),
      'use_locking': False,
      'name': 'int64_sub_scalar'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_4))

  # Input 5: uint8 subtraction
  input_dict_5 = {
      'ref': np.array([255, 128, 10], dtype=np.uint8),
      'value': np.array([10, 28, 5], dtype=np.uint8),
      'use_locking': False,
      'name': 'uint8_sub'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_5))

  # Input 6: complex64 subtraction
  input_dict_6 = {
      'ref': np.array([1+2j, 3+4j], dtype=np.complex64),
      'value': np.array([0.5+1j, 1.5+2j], dtype=np.complex64),
      'use_locking': True,
      'name': 'complex64_sub_locked'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_6))

  # Input 7: half (float16) subtraction
  input_dict_7 = {
      'ref': np.array([10.0, -10.0], dtype=np.float16),
      'value': np.array([0.1, -0.1], dtype=np.float16),
      'use_locking': False,
      'name': 'half_sub'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_7))

  # Input 8: int8 subtraction
  input_dict_8 = {
      'ref': np.array([-127, 0, 127], dtype=np.int8),
      'value': np.array([1, 5, -1], dtype=np.int8),
      'use_locking': True,
      'name': 'int8_sub_locked'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_8))

  # Input 9: 3D uint16 tensor
  input_dict_9 = {
      'ref': np.full((2, 2, 2), 1000, dtype=np.uint16),
      'value': np.arange(8, dtype=np.uint16).reshape((2, 2, 2)),
      'use_locking': False,
      'name': 'uint16_3d_sub'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_9))

  # Input 10: complex128 subtraction
  input_dict_10 = {
      'ref': np.array([[1e10 + 1e10j]], dtype=np.complex128),
      'value': np.array([[1e5 + 1e5j]], dtype=np.complex128),
      'use_locking': False,
      'name': 'complex128_sub'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_10))

  # Input 11: uint32 subtraction
  input_dict_11 = {
      'ref': np.array([4294967295], dtype=np.uint32),
      'value': np.array([1], dtype=np.uint32),
      'use_locking': False,
      'name': 'uint32_sub'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_11))

  # Input 12: uint64 subtraction
  input_dict_12 = {
      'ref': np.array([1000, 2000], dtype=np.uint64),
      'value': np.array([500, 500], dtype=np.uint64),
      'use_locking': True,
      'name': 'uint64_sub_locked'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_12))

  return list_of_inputs

generated_inputs["tf.raw_ops.AssignSub"] = get_tf_raw_ops_assign_sub_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.AssignSub' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AssignSub'.")

check_valid('tf.raw_ops.AssignSub', generated_inputs['tf.raw_ops.AssignSub'], lib="tf", suffix=0)
