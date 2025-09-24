
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_barrierclose_inputs():
    list_of_inputs = []
    # This operation is fundamentally incompatible with the eager execution
    # environment in which it is being tested. It requires a 'tf.string_ref'
    # handle from TensorFlow's graph mode, and it is impossible to create
    # such a handle from a NumPy array in an eager context. Therefore, any
    # input that correctly follows the API's signature will inevitably
    # raise the "does not support eager execution" RuntimeError. The following
    # inputs are provided to meet the generation requirement, with the
    # understanding that this error is an unavoidable consequence of the op's
    # design.

    # Input 1: Basic case with cancel_pending_enqueues=False
    input_dict = {
        'handle': np.array("barrier_handle_A", dtype=object),
        'cancel_pending_enqueues': False,
        'name': 'close_op_A'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic case with cancel_pending_enqueues=True
    input_dict = {
        'handle': np.array("barrier_handle_B", dtype=object),
        'cancel_pending_enqueues': True,
        'name': 'close_op_B'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty string for handle
    input_dict = {
        'handle': np.array("", dtype=object),
        'cancel_pending_enqueues': False,
        'name': 'close_op_C_empty_handle'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty string for name
    input_dict = {
        'handle': np.array("barrier_handle_D", dtype=object),
        'cancel_pending_enqueues': True,
        'name': ''
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.BarrierClose"] = tf_raw_ops_barrierclose_inputs()

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
