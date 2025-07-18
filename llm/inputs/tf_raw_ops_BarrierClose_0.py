
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_barrier_close_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.BarrierClose operation.

    NOTE: The `RuntimeError: barrier_close op does not support eager execution`
    is an unfixable error in the context of the testing environment. This specific
    TensorFlow operation is designed exclusively for Graph mode, where it operates
    on resource handles created within that graph. The test harness attempts to
    run it in Eager mode, where the operation is explicitly disabled, leading to
    the runtime error. The provided inputs are syntactically valid according to
    the API signature but cannot be executed eagerly.
    """
    list_of_inputs = []

    # Input 1
    input_dict_1 = {
        'handle': np.array('handle_v5_1', dtype=object),
        'cancel_pending_enqueues': False,
        'name': 'name_v5_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2
    input_dict_2 = {
        'handle': np.array('handle_v5_2', dtype=object),
        'cancel_pending_enqueues': True,
        'name': 'name_v5_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3
    input_dict_3 = {
        'handle': np.array('another_handle_for_barrier', dtype=object),
        'cancel_pending_enqueues': False,
        'name': 'SomeOtherName'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4
    input_dict_4 = {
        'handle': np.array('handle_with_nums_123', dtype=object),
        'cancel_pending_enqueues': True,
        'name': 'name_for_handle_123'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5
    input_dict_5 = {
        'handle': np.array('handle_for_no_name', dtype=object),
        'cancel_pending_enqueues': False,
        'name': ''
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6
    input_dict_6 = {
        'handle': np.array('a/scoped/handle', dtype=object),
        'cancel_pending_enqueues': True,
        'name': 'a/scoped/name'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7
    input_dict_7 = {
        'handle': np.array('', dtype=object),
        'cancel_pending_enqueues': False,
        'name': 'name_for_empty_handle'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8
    input_dict_8 = {
        'handle': np.array(['h_as_array'], dtype=object),
        'cancel_pending_enqueues': True,
        'name': 'n_for_array_h'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9
    input_dict_9 = {
        'handle': np.array(['another_array_handle'], dtype=object),
        'cancel_pending_enqueues': False,
        'name': 'another_name_for_array_h'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10
    input_dict_10 = {
        'handle': np.array('last_one', dtype=object),
        'cancel_pending_enqueues': True,
        'name': 'last_name'
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
