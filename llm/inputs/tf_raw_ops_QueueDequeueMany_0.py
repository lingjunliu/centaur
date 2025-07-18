
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_raw_ops_queue_dequeue_many_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.QueueDequeueMany function.
    
    NOTE: This operation is not compatible with eager execution and will raise a
    RuntimeError when executed in that context. The inputs provided are structurally
    and type-wise valid according to the API's signature, but their execution
    is expected to fail in a standard eager TensorFlow environment.
    """
    list_of_inputs = []

    # Input 1: Basic case with a single float component.
    input_dict_1 = {
        'handle': np.array("queue_handle_1", dtype=object),
        'n': np.array(5, dtype=np.int32),
        'component_types': [tf.float32],
        'timeout_ms': -1,
        'name': "dequeue_basic"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Multiple component types (int, string, float) and a timeout.
    input_dict_2 = {
        'handle': np.array("queue_handle_2", dtype=object),
        'n': np.array(10, dtype=np.int32),
        'component_types': [tf.int32, tf.string, tf.float64],
        'timeout_ms': 1000,
        'name': "dequeue_multi_type"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Dequeue a single tuple with complex and bool types, zero timeout.
    input_dict_3 = {
        'handle': np.array("queue_handle_3", dtype=object),
        'n': np.array(1, dtype=np.int32),
        'component_types': [tf.complex64, tf.bool],
        'timeout_ms': 0,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Dequeue a large number of tuples.
    input_dict_4 = {
        'handle': np.array("queue_handle_4", dtype=object),
        'n': np.array(100, dtype=np.int32),
        'component_types': [tf.uint8, tf.int16],
        'timeout_ms': -1,
        'name': "dequeue_large_n"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Many different numeric component types.
    input_dict_5 = {
        'handle': np.array("queue_handle_5", dtype=object),
        'n': np.array(2, dtype=np.int32),
        'component_types': [tf.float16, tf.int8, tf.uint16, tf.int64],
        'timeout_ms': 500,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Complex types with a name.
    input_dict_6 = {
        'handle': np.array("queue_handle_6", dtype=object),
        'n': np.array(8, dtype=np.int32),
        'component_types': [tf.complex64, tf.complex128],
        'timeout_ms': -1,
        'name': "dequeue_complex_op"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Single string component with a long timeout.
    input_dict_7 = {
        'handle': np.array("queue_handle_7", dtype=object),
        'n': np.array(20, dtype=np.int32),
        'component_types': [tf.string],
        'timeout_ms': 20000,
        'name': "dequeue_strings_long_wait"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: A long list of identical component types.
    input_dict_8 = {
        'handle': np.array("queue_handle_8", dtype=object),
        'n': np.array(3, dtype=np.int32),
        'component_types': [tf.int32] * 10,
        'timeout_ms': -1,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: bfloat16 type.
    input_dict_9 = {
        'handle': np.array("queue_handle_9", dtype=object),
        'n': np.array(4, dtype=np.int32),
        'component_types': [tf.bfloat16, tf.float64],
        'timeout_ms': 100,
        'name': "dequeue_bfloat"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: n tensor as a 1-D array (often acceptable for scalar inputs).
    input_dict_10 = {
        'handle': np.array("queue_handle_10", dtype=object),
        'n': np.array([12], dtype=np.int32),
        'component_types': [tf.int64, tf.bool],
        'timeout_ms': -1,
        'name': "dequeue_1d_n"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.QueueDequeueMany"] = get_tf_raw_ops_queue_dequeue_many_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QueueDequeueMany' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QueueDequeueMany'.")

check_valid('tf.raw_ops.QueueDequeueMany', generated_inputs['tf.raw_ops.QueueDequeueMany'], lib="tf", suffix=0)
