
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_queuedequeuemany_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.QueueDequeueMany function.
    NOTE: This op is not supported in eager execution. The generated inputs use n=0
    and a placeholder handle, as creating a real queue is not possible with numpy
    and n > 0 will cause a runtime error in eager mode.
    """
    list_of_inputs = []

    # Input 1: Basic case with n=0
    input_dict_1 = {
        'handle': np.array("h1", dtype=object),
        'n': np.array(0, dtype=np.int32),
        'component_types': [np.float32],
        'timeout_ms': -1,
        'name': 'dequeue_zero_float'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Multiple component types
    input_dict_2 = {
        'handle': np.array("h2", dtype=object),
        'n': np.array(0, dtype=np.int32),
        'component_types': [np.float64, np.int32],
        'timeout_ms': -1,
        'name': 'dequeue_zero_multi'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: With a timeout specified
    input_dict_3 = {
        'handle': np.array("h3", dtype=object),
        'n': np.array(0, dtype=np.int32),
        'component_types': [np.int64],
        'timeout_ms': 1000,
        'name': 'dequeue_zero_timeout'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Complex component type
    input_dict_4 = {
        'handle': np.array("h4", dtype=object),
        'n': np.array(0, dtype=np.int32),
        'component_types': [np.complex64],
        'timeout_ms': -1,
        'name': 'dequeue_zero_complex64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Zero timeout
    input_dict_5 = {
        'handle': np.array("h5", dtype=object),
        'n': np.array(0, dtype=np.int32),
        'component_types': [np.int16, np.uint8],
        'timeout_ms': 0,
        'name': 'dequeue_zero_zerotimeout'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: String component type
    input_dict_6 = {
        'handle': np.array("h6", dtype=object),
        'n': np.array(0, dtype=np.int32),
        'component_types': [np.string_],
        'timeout_ms': -1,
        'name': 'dequeue_zero_string'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Boolean component type
    input_dict_7 = {
        'handle': np.array("h7", dtype=object),
        'n': np.array(0, dtype=np.int32),
        'component_types': [np.bool_],
        'timeout_ms': 500,
        'name': 'dequeue_zero_bool'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Complex128 component type
    input_dict_8 = {
        'handle': np.array("h8", dtype=object),
        'n': np.array(0, dtype=np.int32),
        'component_types': [np.complex128],
        'timeout_ms': -1,
        'name': 'dequeue_zero_complex128'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Long list of component types
    input_dict_9 = {
        'handle': np.array("h9", dtype=object),
        'n': np.array(0, dtype=np.int32),
        'component_types': [np.float32, np.int64, np.bool_, np.complex64],
        'timeout_ms': 10,
        'name': 'dequeue_zero_long_list'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Unsigned integer types
    input_dict_10 = {
        'handle': np.array("h10", dtype=object),
        'n': np.array(0, dtype=np.int32),
        'component_types': [np.uint8, np.uint16, np.uint32, np.uint64],
        'timeout_ms': -1,
        'name': 'dequeue_zero_unsigned'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.QueueDequeueMany"] = get_tf_raw_ops_queuedequeuemany_inputs()

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
