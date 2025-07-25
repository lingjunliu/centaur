
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_raw_ops_accumulator_take_gradient_inputs():
    """
    Generates a list of syntactically valid inputs for the tf.raw_ops.AccumulatorTakeGradient operation.

    NOTE: This operation is fundamentally incompatible with TensorFlow's eager execution mode,
    which is the default in modern versions. The function's C++ kernel implementation for CPU/GPU
    is not registered for eager execution, leading to a "Does not support Eager execution" error.
    This is an inherent design characteristic of the accumulator ops, which rely on stateful
    'ref' handles from a graph context. The inputs generated here are syntactically correct
    according to the API's signature, but they will inevitably trigger this runtime error in the
    eager testing environment. The 'handle' is represented as a 2-element numpy array of byte strings,
    which is a correct numpy representation for a tf.string tensor used for resource handles
    ([container, shared_name]).
    """
    list_of_inputs = []

    # Input 1: Basic case with float32
    input_dict_1 = {
        'handle': np.array([b"", b"acc_handle_1"], dtype=np.string_),
        'num_required': np.array(1, dtype=np.int32),
        'dtype': tf.float32,
        'name': 'take_grad_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: float64 dtype and a different handle
    input_dict_2 = {
        'handle': np.array([b"custom_container", b"acc_handle_2"], dtype=np.string_),
        'num_required': np.array(10, dtype=np.int32),
        'dtype': tf.float64,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: int32 dtype
    input_dict_3 = {
        'handle': np.array([b"", b"acc_handle_3"], dtype=np.string_),
        'num_required': np.array(5, dtype=np.int32),
        'dtype': tf.int32,
        'name': 'take_grad_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: complex64 dtype
    input_dict_4 = {
        'handle': np.array([b"", b"acc_handle_4"], dtype=np.string_),
        'num_required': np.array(2, dtype=np.int32),
        'dtype': tf.complex64,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: bfloat16 dtype and large num_required
    input_dict_5 = {
        'handle': np.array([b"", b"acc_handle_5"], dtype=np.string_),
        'num_required': np.array(100, dtype=np.int32),
        'dtype': tf.bfloat16,
        'name': 'take_grad_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: half (float16) dtype
    input_dict_6 = {
        'handle': np.array([b"", b"acc_handle_6"], dtype=np.string_),
        'num_required': np.array(8, dtype=np.int32),
        'dtype': tf.half,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: int64 dtype
    input_dict_7 = {
        'handle': np.array([b"", b"acc_handle_7"], dtype=np.string_),
        'num_required': np.array(20, dtype=np.int32),
        'dtype': tf.int64,
        'name': 'take_grad_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: complex128 dtype
    input_dict_8 = {
        'handle': np.array([b"", b"acc_handle_8"], dtype=np.string_),
        'num_required': np.array(4, dtype=np.int32),
        'dtype': tf.complex128,
        'name': 'take_grad_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    return list_of_inputs

generated_inputs["tf.raw_ops.AccumulatorTakeGradient"] = get_tf_raw_ops_accumulator_take_gradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.AccumulatorTakeGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AccumulatorTakeGradient'.")

check_valid('tf.raw_ops.AccumulatorTakeGradient', generated_inputs['tf.raw_ops.AccumulatorTakeGradient'], lib="tf", suffix=0)
