
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_delete_session_tensor_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    handle_val = b"tensor_handle_1"
    input_dict = {
        "handle": np.array(handle_val, dtype=np.string_),
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different tensor handle
    handle_val = b"another_tensor_handle"
    input_dict = {
        "handle": np.array(handle_val, dtype=np.string_),
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty name
    handle_val = b"empty_name_handle"
    input_dict = {
        "handle": np.array(handle_val, dtype=np.string_),
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Handle with numbers
    handle_val = b"tensor_handle_123"
    input_dict = {
        "handle": np.array(handle_val, dtype=np.string_),
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Long handle
    handle_val = b"this_is_a_very_long_tensor_handle_string"
    input_dict = {
        "handle": np.array(handle_val, dtype=np.string_),
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Handle with special characters
    handle_val = b"tensor_handle!@#$"
    input_dict = {
        "handle": np.array(handle_val, dtype=np.string_),
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different name
    handle_val = b"handle_7"
    input_dict = {
        "handle": np.array(handle_val, dtype=np.string_),
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Another long name
    handle_val = b"handle_8"
    input_dict = {
        "handle": np.array(handle_val, dtype=np.string_),
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Name with numbers
    handle_val = b"handle_9"
    input_dict = {
        "handle": np.array(handle_val, dtype=np.string_),
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Name with special characters
    handle_val = b"handle_10"
    input_dict = {
        "handle": np.array(handle_val, dtype=np.string_),
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Name same as handle
    handle_val = b"same_name_handle"
    input_dict = {
        "handle": np.array(handle_val, dtype=np.string_),
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Short handle
    handle_val = b"h1"
    input_dict = {
        "handle": np.array(handle_val, dtype=np.string_),
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DeleteSessionTensor"] = tf_raw_ops_delete_session_tensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DeleteSessionTensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DeleteSessionTensor'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.DeleteSessionTensor', generated_inputs['tf.raw_ops.DeleteSessionTensor'], lib="tf", suffix=0)
