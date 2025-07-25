
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_lmdbreader_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.LMDBReader operation.
    NOTE: This operation is only compatible with TensorFlow's graph execution mode.
    The test harness may fail if it attempts to eagerly convert the symbolic output tensor
    to a numpy array without running a session.
    """
    try:
        if tf.executing_eagerly():
            tf.compat.v1.disable_eager_execution()
    except (AttributeError, RuntimeError):
        # Eager execution might already be disabled.
        pass

    list_of_inputs = []

    # Input 1: Default empty strings
    input_dict = {
        'container': '',
        'shared_name': '',
        'name': 'reader_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Non-empty container
    input_dict = {
        'container': 'my_container',
        'shared_name': '',
        'name': 'reader_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Non-empty shared_name
    input_dict = {
        'container': '',
        'shared_name': 'my_shared_reader',
        'name': 'reader_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Both container and shared_name are non-empty
    input_dict = {
        'container': 'container_alpha',
        'shared_name': 'shared_reader_alpha',
        'name': 'reader_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Using underscores
    input_dict = {
        'container': 'test_container_v1',
        'shared_name': 'test_shared_name_v1',
        'name': 'reader_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Using numbers
    input_dict = {
        'container': 'container123',
        'shared_name': 'sharedname456',
        'name': 'reader_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Long string values
    input_dict = {
        'container': 'a_long_but_valid_container_name_for_the_reader',
        'shared_name': 'a_long_but_valid_shared_name_for_the_reader',
        'name': 'reader_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Another combination of non-empty and empty strings
    input_dict = {
        'container': 'some_other_container',
        'shared_name': '',
        'name': 'reader_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Another combination with shared name
    input_dict = {
        'container': '',
        'shared_name': 'another_shared_reader_instance',
        'name': 'reader_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Uppercase characters
    input_dict = {
        'container': 'ContainerWithUpperCase',
        'shared_name': 'SharedNameWithUpperCase',
        'name': 'Reader_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.LMDBReader"] = tf_raw_ops_lmdbreader_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.LMDBReader' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.LMDBReader'.")

check_valid('tf.raw_ops.LMDBReader', generated_inputs['tf.raw_ops.LMDBReader'], lib="tf", suffix=0)
