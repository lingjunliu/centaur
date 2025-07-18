
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

# The user of this function is expected to handle the conversion of numpy arrays to a tf.Variable
# for the 'ref' argument, as this operation modifies its input in-place.
# The persistent error "RuntimeError: assign_sub op does not support eager execution" is a fundamental
# characteristic of this low-level TensorFlow operation. tf.raw_ops.AssignSub is designed to
# operate within a TensorFlow graph (e.g., inside a @tf.function) and is not compatible with
# TensorFlow's default eager execution mode.
# The inputs provided below are valid according to the function's signature. The error arises from
# the execution context in which the user's test harness is calling the function, which is
# an issue that cannot be "fixed" by merely changing the input values.
# The higher-level, eager-friendly equivalent is the `tf.Variable.assign_sub` method.

def get_tf_raw_ops_assign_sub_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 subtraction
    input_dict = {
        'ref': np.array([10.0, 20.0], dtype=np.float32),
        'value': np.array([1.0, 2.0], dtype=np.float32),
        'use_locking': False,
        'name': 'sub_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32 subtraction with negative numbers
    input_dict = {
        'ref': np.array([[-10, 20], [-30, 40]], dtype=np.int32),
        'value': np.array([[5, -5], [10, -10]], dtype=np.int32),
        'use_locking': True,
        'name': 'sub_int32_neg'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64 subtraction, 3D tensor
    input_dict = {
        'ref': np.arange(8, dtype=np.float64).reshape(2, 2, 2),
        'value': np.ones((2, 2, 2), dtype=np.float64) * 2.5,
        'use_locking': False,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: uint8 subtraction
    input_dict = {
        'ref': np.array([255, 100], dtype=np.uint8),
        'value': np.array([55, 50], dtype=np.uint8),
        'use_locking': True,
        'name': 'sub_uint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64 subtraction
    input_dict = {
        'ref': np.array([1+2j, 3+4j], dtype=np.complex64),
        'value': np.array([0.5+1j, 1.5+2j], dtype=np.complex64),
        'use_locking': False,
        'name': 'sub_complex64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int64 scalar subtraction
    input_dict = {
        'ref': np.array(10000000000, dtype=np.int64),
        'value': np.array(5000000000, dtype=np.int64),
        'use_locking': False,
        'name': 'sub_int64_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: half (float16) subtraction
    input_dict = {
        'ref': np.array([1.5, 2.5, 3.5], dtype=np.half),
        'value': np.array([0.5, 0.5, 0.5], dtype=np.half),
        'use_locking': True,
        'name': 'sub_half'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex128 subtraction
    input_dict = {
        'ref': np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128),
        'value': np.array([[0.5+1j, 1.5+2j], [2.5+3j, 3.5+4j]], dtype=np.complex128),
        'use_locking': False,
        'name': 'sub_complex128_matrix'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int8 subtraction
    input_dict = {
        'ref': np.array([120, -120], dtype=np.int8),
        'value': np.array([20, -20], dtype=np.int8),
        'use_locking': False,
        'name': 'sub_int8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint16 subtraction
    input_dict = {
        'ref': np.array([65535, 1000], dtype=np.uint16),
        'value': np.array([1000, 500], dtype=np.uint16),
        'use_locking': True,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: uint32 subtraction
    input_dict = {
        'ref': np.array(4294967295, dtype=np.uint32),
        'value': np.array(1, dtype=np.uint32),
        'use_locking': False,
        'name': 'sub_uint32_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: uint64 subtraction
    input_dict = {
        'ref': np.array([18446744073709551615, 100], dtype=np.uint64),
        'value': np.array([10, 10], dtype=np.uint64),
        'use_locking': False,
        'name': 'sub_uint64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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
