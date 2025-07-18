
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_get_session_tensor_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.GetSessionTensor function.
    The 'handle' inputs are placeholders. This op will raise a FailedPreconditionError
    if not used within a context where the handle has been previously created and stored.
    The inputs provided are syntactically and type-correct according to the signature.
    """
    list_of_inputs = []

    # Input 1
    input_dict = {
        'handle': np.array('handle_float32', dtype=object),
        'dtype': np.float32,
        'name': "get_tensor_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'handle': np.array('handle_float64', dtype=object),
        'dtype': np.float64,
        'name': "get_tensor_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'handle': np.array('handle_int32', dtype=object),
        'dtype': np.int32,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'handle': np.array('handle_int64', dtype=object),
        'dtype': np.int64,
        'name': "get_tensor_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'handle': np.array('handle_uint8', dtype=object),
        'dtype': np.uint8,
        'name': "get_tensor_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'handle': np.array('handle_int16', dtype=object),
        'dtype': np.int16,
        'name': "get_tensor_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'handle': np.array('handle_bool', dtype=object),
        'dtype': np.bool_,
        'name': "get_tensor_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'handle': np.array('handle_complex64', dtype=object),
        'dtype': np.complex64,
        'name': "get_tensor_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'handle': np.array('handle_complex128', dtype=object),
        'dtype': np.complex128,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'handle': np.array('another_handle', dtype=object),
        'dtype': np.float32,
        'name': "get_tensor_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_dict = {
        'handle': np.array('string_tensor_handle', dtype=object),
        'dtype': np.dtype('O'),
        'name': "get_string_tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_dict = {
        'handle': np.array('handle_float16', dtype=object),
        'dtype': np.float16,
        'name': "get_tensor_12"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.GetSessionTensor"] = tf_raw_ops_get_session_tensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.GetSessionTensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.GetSessionTensor'.")

check_valid('tf.raw_ops.GetSessionTensor', generated_inputs['tf.raw_ops.GetSessionTensor'], lib="tf", suffix=0)
