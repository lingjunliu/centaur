
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_DeleteSessionTensor_inputs():
    """
    Generates a list of syntactically valid inputs for tf.raw_ops.DeleteSessionTensor.
    This operation is stateful and requires a valid tensor handle from an active
    session. Calling it with static strings in a stateless context will
    inherently lead to a FailedPreconditionError at runtime because the handle
    does not exist. The provided inputs are valid according to the function's signature.
    """
    list_of_inputs = []

    # Input 1
    input_dict_1 = {
        'handle': 'a',
        'name': 'delete_a'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2
    input_dict_2 = {
        'handle': 'b'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3
    input_dict_3 = {
        'handle': 'c_1',
        'name': 'delete_c_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4
    input_dict_4 = {
        'handle': 'd_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Input 5
    input_dict_5 = {
        'handle': 'MyHandle',
        'name': 'DeleteMyHandle'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6
    input_dict_6 = {
        'handle': 'AnotherHandle'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    return list_of_inputs

generated_inputs["tf.raw_ops.DeleteSessionTensor"] = get_tf_raw_ops_DeleteSessionTensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DeleteSessionTensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DeleteSessionTensor'.")

check_valid('tf.raw_ops.DeleteSessionTensor', generated_inputs['tf.raw_ops.DeleteSessionTensor'], lib="tf", suffix=0)
