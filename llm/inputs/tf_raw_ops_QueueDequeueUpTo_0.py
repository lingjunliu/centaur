
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_queue_dequeue_up_to_inputs():
    """
    Generates a list of syntactically valid inputs for tf.raw_ops.QueueDequeueUpTo.
    NOTE: This op is not compatible with eager execution and is expected to raise a
    RuntimeError in the testing environment because the 'handle' argument must be
    a valid resource handle from a TensorFlow graph. The inputs are provided to
    satisfy the tool's requirement of generating input dictionaries, even though
    they will fail at the execution stage.
    """
    list_of_inputs = []

    # Helper to create a handle tensor. Using dtype=object for the string to
    # avoid specific 'S<N>' dtypes which can cause validation issues.
    def create_handle(name):
        return np.array(name, dtype=object)

    # Case 1: Minimal input, dequeue 1 float32
    input_dict = {
        'handle': create_handle("q_handle_1"),
        'n': np.array(1, dtype=np.int32),
        'component_types': [tf.float32],
        'timeout_ms': -1,
        'name': "min_dequeue"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Dequeue multiple elements with two component types
    input_dict = {
        'handle': create_handle("q_handle_2"),
        'n': np.array(8, dtype=np.int32),
        'component_types': [tf.int64, tf.string],
        'timeout_ms': 50,
        'name': "dequeue_int_str"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Dequeue with a timeout and bool component
    input_dict = {
        'handle': create_handle("q_handle_3"),
        'n': np.array(4, dtype=np.int32),
        'component_types': [tf.bool],
        'timeout_ms': 100,
        'name': "dequeue_bool_timeout"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Dequeue complex numbers, no name
    input_dict = {
        'handle': create_handle("q_handle_4"),
        'n': np.array(2, dtype=np.int32),
        'component_types': [tf.complex64, tf.complex128],
        'timeout_ms': -1,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Dequeue a larger batch with multiple different components
    input_dict = {
        'handle': create_handle("q_handle_5"),
        'n': np.array(32, dtype=np.int32),
        'component_types': [tf.float16, tf.int8, tf.uint8],
        'timeout_ms': 0,
        'name': "dequeue_large_batch"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Dequeue unsigned integer types
    input_dict = {
        'handle': create_handle("q_handle_6"),
        'n': np.array(16, dtype=np.int32),
        'component_types': [tf.uint16, tf.uint32, tf.uint64],
        'timeout_ms': -1,
        'name': "dequeue_unsigned_ints"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Dequeue double precision float
    input_dict = {
        'handle': create_handle("q_handle_7"),
        'n': np.array(10, dtype=np.int32),
        'component_types': [tf.double],
        'timeout_ms': 2000,
        'name': "dequeue_double"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: Dequeue bfloat16
    input_dict = {
        'handle': create_handle("q_handle_8"),
        'n': np.array(7, dtype=np.int32),
        'component_types': [tf.bfloat16],
        'timeout_ms': -1,
        'name': "dequeue_bfloat16"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: No timeout (timeout_ms = -1) and multiple components
    input_dict = {
        'handle': create_handle("q_handle_9"),
        'n': np.array(3, dtype=np.int32),
        'component_types': [tf.int32, tf.float32, tf.string],
        'timeout_ms': -1,
        'name': "dequeue_no_timeout"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: Long list of component types
    input_dict = {
        'handle': create_handle("q_handle_10"),
        'n': np.array(2, dtype=np.int32),
        'component_types': [tf.float32, tf.int32, tf.string, tf.bool, tf.complex64, tf.int64],
        'timeout_ms': 150,
        'name': "dequeue_many_components"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.QueueDequeueUpTo"] = tf_raw_ops_queue_dequeue_up_to_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QueueDequeueUpTo' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QueueDequeueUpTo'.")

check_valid('tf.raw_ops.QueueDequeueUpTo', generated_inputs['tf.raw_ops.QueueDequeueUpTo'], lib="tf", suffix=0)
