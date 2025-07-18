
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_queue_dequeue_inputs():
    """
    Generates a list of syntactically valid inputs for the tf.raw_ops.QueueDequeue function.
    NOTE: This op is fundamentally incompatible with eager execution. The test harness
    is expected to encounter a RuntimeError, as the op is designed for graph mode only.
    The inputs themselves are valid according to the function signature.
    """
    list_of_inputs = []

    # Input 1: Basic case with a single float32 type.
    input_dict_1 = {
        'handle': np.array("queue_handle_float32", dtype=object),
        'component_types': [tf.float32.as_numpy_dtype],
        'timeout_ms': -1,
        'name': 'dequeue_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic case with a single int32 type.
    input_dict_2 = {
        'handle': np.array("queue_handle_int32", dtype=object),
        'component_types': [tf.int32.as_numpy_dtype],
        'timeout_ms': 0,
        'name': 'dequeue_int32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Basic case with a single string type.
    input_dict_3 = {
        'handle': np.array("queue_handle_string", dtype=object),
        'component_types': [tf.string.as_numpy_dtype],
        'timeout_ms': 1000,
        'name': 'dequeue_string'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Tuple of (int64, float64).
    input_dict_4 = {
        'handle': np.array("queue_handle_tuple1", dtype=object),
        'component_types': [tf.int64.as_numpy_dtype, tf.float64.as_numpy_dtype],
        'timeout_ms': -1,
        'name': 'dequeue_int_float_tuple'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Tuple of (bool, string).
    input_dict_5 = {
        'handle': np.array("queue_handle_tuple2", dtype=object),
        'component_types': [tf.bool.as_numpy_dtype, tf.string.as_numpy_dtype],
        'timeout_ms': 50,
        'name': 'dequeue_bool_string_tuple'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Complex64 type.
    input_dict_6 = {
        'handle': np.array("queue_handle_complex64", dtype=object),
        'component_types': [tf.complex64.as_numpy_dtype],
        'timeout_ms': -1,
        'name': 'dequeue_complex64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Complex128 type.
    input_dict_7 = {
        'handle': np.array("queue_handle_complex128", dtype=object),
        'component_types': [tf.complex128.as_numpy_dtype],
        'timeout_ms': 20,
        'name': 'dequeue_complex128'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Unsigned integer type.
    input_dict_8 = {
        'handle': np.array("queue_handle_uint16", dtype=object),
        'component_types': [tf.uint16.as_numpy_dtype],
        'timeout_ms': -1,
        'name': 'dequeue_uint16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Half-precision float (float16).
    input_dict_9 = {
        'handle': np.array("queue_handle_float16", dtype=object),
        'component_types': [tf.float16.as_numpy_dtype],
        'timeout_ms': 1,
        'name': 'dequeue_float16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: bfloat16 type.
    input_dict_10 = {
        'handle': np.array("queue_handle_bfloat16", dtype=object),
        'component_types': [tf.bfloat16.as_numpy_dtype],
        'timeout_ms': -1,
        'name': 'dequeue_bfloat16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.QueueDequeue"] = tf_raw_ops_queue_dequeue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QueueDequeue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QueueDequeue'.")

check_valid('tf.raw_ops.QueueDequeue', generated_inputs['tf.raw_ops.QueueDequeue'], lib="tf", suffix=0)
