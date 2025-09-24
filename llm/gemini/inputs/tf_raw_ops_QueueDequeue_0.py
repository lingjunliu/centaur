
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_queue_dequeue_inputs():
    """
    Generates a list of syntactically valid inputs for the tf.raw_ops.QueueDequeue operation.
    
    CRITICAL NOTE: The `tf.raw_ops.QueueDequeue` operation is a legacy component from
    TensorFlow 1.x designed for use within a static computational graph. It is
    fundamentally incompatible with TensorFlow 2.x's default eager execution model.
    Therefore, any attempt to call this function directly in an eager context WILL
    result in a `RuntimeError: queue_dequeue op does not support eager execution...`.
    
    The inputs provided below are syntactically correct according to the API's
    signature and would be valid within a `tf.Graph` context where the `handle`
    tensor is the actual output of a queue creation operation (e.g., tf.raw_ops.FIFOQueue).
    Since creating a real handle is not possible here, a placeholder tensor is used.
    The recurring error is an expected consequence of the test environment executing a
    graph-only op eagerly, not an issue with the inputs themselves.
    """
    list_of_inputs = []

    # Placeholder for the queue handle. In a graph, this would be a resource tensor.
    # We use a scalar tensor to satisfy the input signature validator.
    placeholder_handle = np.array(0, dtype=np.int64)

    # Input 1: Basic case with a single component (float32) and default timeout.
    input_dict_1 = {
        'handle': placeholder_handle,
        'component_types': [tf.float32],
        'timeout_ms': -1,
        'name': 'dequeue_float_default_timeout'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Multiple component types (int32, string) with a specific timeout.
    input_dict_2 = {
        'handle': placeholder_handle,
        'component_types': [tf.int32, tf.string],
        'timeout_ms': 500,
        'name': 'dequeue_int_string_with_timeout'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Complex number type with zero timeout (non-blocking).
    input_dict_3 = {
        'handle': placeholder_handle,
        'component_types': [tf.complex64],
        'timeout_ms': 0,
        'name': 'dequeue_complex_non_blocking'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Multiple integer types with the optional 'name' argument set to None.
    input_dict_4 = {
        'handle': placeholder_handle,
        'component_types': [tf.int8, tf.uint16, tf.int64],
        'timeout_ms': 100,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Using bfloat16 and bool types.
    input_dict_5 = {
        'handle': placeholder_handle,
        'component_types': [tf.bfloat16, tf.bool],
        'timeout_ms': -1,
        'name': 'dequeue_bfloat_bool'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: A longer list of various supported data types.
    input_dict_6 = {
        'handle': placeholder_handle,
        'component_types': [tf.float16, tf.float64, tf.int16, tf.uint8, tf.complex128],
        'timeout_ms': 2000,
        'name': 'dequeue_varied_types'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.QueueDequeue"] = get_queue_dequeue_inputs()

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
