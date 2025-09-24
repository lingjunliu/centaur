
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_barrier_incomplete_size_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.BarrierIncompleteSize operation.

    This operation is designed for TensorFlow's graph mode and is expected to
    raise a RuntimeError in eager execution, as the 'handle' argument is a reference
    to a stateful resource. The provided inputs are syntactically valid according
    to the API signature, using numpy arrays with dtype=object to represent the
    string tensor 'handle' to ensure compatibility with the testing framework's dtype validation.
    """
    list_of_inputs = []

    # Input 1: Basic valid input
    input_1 = {
        'handle': np.array('barrier_handle_1', dtype=object),
        'name': 'test_name_1'
    }
    list_of_inputs.append(copy.deepcopy(input_1))

    # Input 2: Another basic valid input
    input_2 = {
        'handle': np.array('another_barrier_handle', dtype=object),
        'name': 'TestName'
    }
    list_of_inputs.append(copy.deepcopy(input_2))

    # Input 3: Name with underscores
    input_3 = {
        'handle': np.array('barrier_123', dtype=object),
        'name': 'my_operation_name'
    }
    list_of_inputs.append(copy.deepcopy(input_3))

    # Input 4: Long handle and name strings
    input_4 = {
        'handle': np.array('a_very_long_and_descriptive_barrier_handle_for_testing_purposes', dtype=object),
        'name': 'AVeryLongAndSpecificOperationNameThatMightTestInternalBufferLimits'
    }
    list_of_inputs.append(copy.deepcopy(input_4))

    # Input 5: Handle and name with special characters
    input_5 = {
        'handle': np.array('handle/with/slashes', dtype=object),
        'name': 'op_name.with-special_chars_123'
    }
    list_of_inputs.append(copy.deepcopy(input_5))

    # Input 6: Name is an empty string
    input_6 = {
        'handle': np.array('some_handle', dtype=object),
        'name': ''
    }
    list_of_inputs.append(copy.deepcopy(input_6))

    # Input 7: Empty string for handle
    input_7 = {
        'handle': np.array('', dtype=object),
        'name': 'empty_handle_test'
    }
    list_of_inputs.append(copy.deepcopy(input_7))

    # Input 8: Numeric-like strings for handle and name
    input_8 = {
        'handle': np.array('1234567890', dtype=object),
        'name': '9876543210'
    }
    list_of_inputs.append(copy.deepcopy(input_8))

    # Input 9: Handle and name with mixed case
    input_9 = {
        'handle': np.array('MixedCaseBarrierHandle', dtype=object),
        'name': 'MixedCaseOperationName'
    }
    list_of_inputs.append(copy.deepcopy(input_9))

    # Input 10: A name that might be interpreted as a scope
    input_10 = {
        'handle': np.array('scoped_barrier_handle', dtype=object),
        'name': 'my_scope/my_op_name'
    }
    list_of_inputs.append(copy.deepcopy(input_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.BarrierIncompleteSize"] = tf_raw_ops_barrier_incomplete_size_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.BarrierIncompleteSize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BarrierIncompleteSize'.")

check_valid('tf.raw_ops.BarrierIncompleteSize', generated_inputs['tf.raw_ops.BarrierIncompleteSize'], lib="tf", suffix=0)
