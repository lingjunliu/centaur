
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_barrier_ready_size_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.BarrierReadySize operation.
    NOTE: This operation is not supported in eager execution mode and is expected
    to raise a RuntimeError. However, to satisfy the testing framework's requirement
    for input generation, syntactically valid inputs are provided.
    """
    list_of_inputs = []

    # The 'handle' parameter must be a tensor. To create a numpy representation
    # of a scalar string tensor that is compatible with the testing framework,
    # we use np.array with dtype=object. This ensures the object has a .shape
    # and a generic .dtype ('O').

    # Input 1: Basic case
    input_dict_1 = {
        'handle': np.array(b"barrier_handle_1", dtype=object),
        'name': 'test_name_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Different handle and name
    input_dict_2 = {
        'handle': np.array(b"another_barrier_handle", dtype=object),
        'name': 'ReadySizeOp'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Name with scope
    input_dict_3 = {
        'handle': np.array(b"scoped/barrier/handle", dtype=object),
        'name': 'my_scope/ready_size'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Empty string for handle
    input_dict_4 = {
        'handle': np.array(b"", dtype=object),
        'name': 'empty_handle_op'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Empty string for name
    input_dict_5 = {
        'handle': np.array(b"handle_for_empty_name", dtype=object),
        'name': ''
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Handle with numbers
    input_dict_6 = {
        'handle': np.array(b"barrier12345", dtype=object),
        'name': 'numeric_name_123'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Handle with special characters
    input_dict_7 = {
        'handle': np.array(b"barrier_!@#$_-", dtype=object),
        'name': 'special_char_name'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Long handle string
    input_dict_8 = {
        'handle': np.array(b"a_very_long_and_specific_barrier_handle_for_testing", dtype=object),
        'name': 'long_handle_op'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Long name string
    input_dict_9 = {
        'handle': np.array(b"short_handle", dtype=object),
        'name': 'a_very_long_and_specific_operation_name_that_is_still_valid'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Single character handle and name
    input_dict_10 = {
        'handle': np.array(b"b", dtype=object),
        'name': 'r'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.BarrierReadySize"] = tf_raw_ops_barrier_ready_size_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.BarrierReadySize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BarrierReadySize'.")

check_valid('tf.raw_ops.BarrierReadySize', generated_inputs['tf.raw_ops.BarrierReadySize'], lib="tf", suffix=0)
