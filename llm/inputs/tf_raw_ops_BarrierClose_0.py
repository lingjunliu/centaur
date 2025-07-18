
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_barrier_close_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.BarrierClose function.
    NOTE: This op is not supported in eager mode and will raise a RuntimeError
    if executed outside of a TensorFlow graph. The inputs provided here are
    syntactically valid for graph-based execution.
    """
    list_of_inputs = []

    # Input 1: Basic case with cancel_pending_enqueues=False
    input_dict_1 = {
        'handle': np.array(b'barrier_handle_A', dtype=np.object_),
        'cancel_pending_enqueues': False,
        'name': 'close_A'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic case with cancel_pending_enqueues=True
    input_dict_2 = {
        'handle': np.array(b'barrier_handle_B', dtype=np.object_),
        'cancel_pending_enqueues': True,
        'name': 'close_B_cancel'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: No optional name
    input_dict_3 = {
        'handle': np.array(b'barrier_handle_C', dtype=np.object_),
        'cancel_pending_enqueues': False,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Handle with numbers
    input_dict_4 = {
        'handle': np.array(b'barrier_123', dtype=np.object_),
        'cancel_pending_enqueues': True,
        'name': 'close_123'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Empty string handle
    input_dict_5 = {
        'handle': np.array(b'', dtype=np.object_),
        'cancel_pending_enqueues': False,
        'name': 'close_empty'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Long handle string
    input_dict_6 = {
        'handle': np.array(b'a_very_long_and_specific_barrier_handle_string_for_testing', dtype=np.object_),
        'cancel_pending_enqueues': False,
        'name': 'close_long_handle'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: Handle as a 2-element array (resource handle convention)
    input_dict_7 = {
        'handle': np.array([b'shared_container', b'resource_barrier_1'], dtype=np.object_),
        'cancel_pending_enqueues': False,
        'name': 'close_resource_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: Another 2-element handle with cancellation
    input_dict_8 = {
        'handle': np.array([b'default', b'resource_barrier_2'], dtype=np.object_),
        'cancel_pending_enqueues': True,
        'name': 'cancel_resource_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Handle as a 1-element array
    input_dict_9 = {
        'handle': np.array([b'single_item_handle'], dtype=np.object_),
        'cancel_pending_enqueues': False,
        'name': 'close_single_item'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: Long operation name
    input_dict_10 = {
        'handle': np.array(b'short_handle', dtype=np.object_),
        'cancel_pending_enqueues': True,
        'name': 'ThisIsAVeryLongAndUnnecessarilyVerboseOperationNameJustForTestingPurposes'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.BarrierClose"] = tf_raw_ops_barrier_close_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.BarrierClose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BarrierClose'.")

check_valid('tf.raw_ops.BarrierClose', generated_inputs['tf.raw_ops.BarrierClose'], lib="tf", suffix=0)
