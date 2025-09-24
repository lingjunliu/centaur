
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_getsessionhandlev2_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.GetSessionHandleV2 function.

    The recurring 'FailedPreconditionError: GetSessionHandle called on null session state'
    is an environmental issue, not an input issue. This raw operation is designed for
    TensorFlow's session-based execution model (e.g., TF1.x), where it stores a tensor
    within an active session. When run in the default eager execution mode of modern
    TensorFlow (TF2.x), no such session exists, causing this unavoidable error. No modification
    of the input numpy array ('value') or the 'name' can fix this fundamental
    incompatibility between the operation and the execution environment. This function
    provides a minimal set of syntactically correct inputs as a best-effort response.
    """
    list_of_inputs = []

    # Input 1: A single, minimal, valid input.
    input_dict_1 = {
        'value': np.array([42.0], dtype=np.float32),
        'name': 'the_only_handle'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))
    
    # Input 2: Another single, minimal, valid input with a different type.
    input_dict_2 = {
        'value': np.array([10], dtype=np.int32),
        'name': 'another_handle'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    return list_of_inputs

generated_inputs["tf.raw_ops.GetSessionHandleV2"] = get_tf_raw_ops_getsessionhandlev2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.GetSessionHandleV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.GetSessionHandleV2'.")

check_valid('tf.raw_ops.GetSessionHandleV2', generated_inputs['tf.raw_ops.GetSessionHandleV2'], lib="tf", suffix=0)
