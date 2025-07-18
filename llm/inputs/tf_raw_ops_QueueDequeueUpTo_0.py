
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_raw_ops_QueueDequeueUpTo_inputs():
    """
    Generates a list of syntactically valid inputs for tf.raw_ops.QueueDequeueUpTo.
    NOTE: This TensorFlow operation is designed for graph mode and is not compatible
    with eager execution. Executing it in an eager context will inherently raise a
    RuntimeError, as the 'handle' argument expects a resource handle from a TF graph.
    The provided inputs conform to the function's signature but will trigger this error
    if run eagerly.
    """
    list_of_inputs = []

    # Input 1: Basic case, float32, default timeout
    list_of_inputs.append({
        'handle': np.array('queue_handle_1', dtype=object),
        'n': np.array(5, dtype=np.int32),
        'component_types': [tf.float32],
        'timeout_ms': -1,
        'name': 'dequeue_floats'
    })

    # Input 2: Two components (int32, string), with a timeout
    list_of_inputs.append({
        'handle': np.array('queue_handle_2', dtype=object),
        'n': np.array(10, dtype=np.int32),
        'component_types': [tf.int32, tf.string],
        'timeout_ms': 500,
        'name': 'dequeue_int_string'
    })

    # Input 3: Dequeue a single tuple
    list_of_inputs.append({
        'handle': np.array('queue_handle_3', dtype=object),
        'n': np.array(1, dtype=np.int32),
        'component_types': [tf.complex64, tf.bool],
        'timeout_ms': -1,
        'name': 'dequeue_single_tuple'
    })

    # Input 4: Multiple component types, no-wait timeout
    list_of_inputs.append({
        'handle': np.array('queue_handle_4', dtype=object),
        'n': np.array(8, dtype=np.int32),
        'component_types': [tf.float16, tf.int8, tf.uint8],
        'timeout_ms': 0,
        'name': 'dequeue_mixed_nowait'
    })

    # Input 5: Large batch dequeue
    list_of_inputs.append({
        'handle': np.array('queue_handle_5', dtype=object),
        'n': np.array(100, dtype=np.int32),
        'component_types': [tf.uint16],
        'timeout_ms': -1,
        'name': 'dequeue_large_batch'
    })

    # Input 6: complex128 type
    list_of_inputs.append({
        'handle': np.array('queue_handle_6', dtype=object),
        'n': np.array(3, dtype=np.int32),
        'component_types': [tf.complex128],
        'timeout_ms': -1,
        'name': 'dequeue_complex128'
    })

    # Input 7: bfloat16 type
    list_of_inputs.append({
        'handle': np.array('queue_handle_7', dtype=object),
        'n': np.array(16, dtype=np.int32),
        'component_types': [tf.bfloat16],
        'timeout_ms': 250,
        'name': 'dequeue_bfloat16'
    })
    
    # Input 8: All float types
    list_of_inputs.append({
        'handle': np.array('queue_handle_8', dtype=object),
        'n': np.array(4, dtype=np.int32),
        'component_types': [tf.float16, tf.bfloat16, tf.float32, tf.float64],
        'timeout_ms': -1,
        'name': 'dequeue_all_floats'
    })

    # Input 9: All standard integer types
    list_of_inputs.append({
        'handle': np.array('queue_handle_9', dtype=object),
        'n': np.array(7, dtype=np.int32),
        'component_types': [tf.int8, tf.uint8, tf.int16, tf.uint16, tf.int32, tf.int64],
        'timeout_ms': -1,
        'name': 'dequeue_all_ints'
    })
    
    # Input 10: No operation name
    list_of_inputs.append({
        'handle': np.array('queue_handle_10', dtype=object),
        'n': np.array(2, dtype=np.int32),
        'component_types': [tf.int32],
        'timeout_ms': -1,
        'name': None
    })

    return [copy.deepcopy(i) for i in list_of_inputs]

generated_inputs["tf.raw_ops.QueueDequeueUpTo"] = get_tf_raw_ops_QueueDequeueUpTo_inputs()

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
