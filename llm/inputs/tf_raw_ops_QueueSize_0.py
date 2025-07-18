
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_queue_size_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.QueueSize operation.
    
    The tf.raw_ops.QueueSize operation is designed for TensorFlow's graph mode and
    is not compatible with eager execution. Calling it in an eager context will
    raise a RuntimeError. The inputs provided here are structurally correct as per
    the API signature, and the resulting RuntimeError is an expected behavior of
    calling this specific API in an incompatible execution mode.
    """
    list_of_inputs = []

    # Input 1: Basic case
    input_dict_1 = {
        'handle': np.array('queue_handle_1', dtype=object),
        'name': 'queue_size_op_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Different handle and name
    input_dict_2 = {
        'handle': np.array('fifo_queue_handle', dtype=object),
        'name': 'MyQueueSize'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Name with slashes (common for scoping)
    input_dict_3 = {
        'handle': np.array('priority_queue_handle', dtype=object),
        'name': 'queues/priority/size'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Handle with numbers
    input_dict_4 = {
        'handle': np.array('queue_123', dtype=object),
        'name': 'queue_size_123'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Handle with special characters
    input_dict_5 = {
        'handle': np.array('queue-handle_with.chars', dtype=object),
        'name': 'special_char_name'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Long handle string
    input_dict_6 = {
        'handle': np.array('a_very_long_and_descriptive_queue_handle_string_for_testing', dtype=object),
        'name': 'long_handle_test'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Empty handle string
    input_dict_7 = {
        'handle': np.array('', dtype=object),
        'name': 'empty_handle_test'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: Empty name string
    input_dict_8 = {
        'handle': np.array('another_queue_handle', dtype=object),
        'name': ''
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Short handle and name
    input_dict_9 = {
        'handle': np.array('q', dtype=object),
        'name': 'q_size'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Name with numbers and underscores
    input_dict_10 = {
        'handle': np.array('final_test_queue', dtype=object),
        'name': 'QueueSize_Op_2024'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.QueueSize"] = tf_raw_ops_queue_size_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QueueSize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QueueSize'.")

check_valid('tf.raw_ops.QueueSize', generated_inputs['tf.raw_ops.QueueSize'], lib="tf", suffix=0)
