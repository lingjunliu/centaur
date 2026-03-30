
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_GetSessionTensor_inputs():
    list_of_inputs = []

    # Input 1: Simple string handle, float32 dtype
    handle = np.array("tensor_handle_1", dtype=np.object_)
    dtype = np.float32
    input_dict = {"handle": handle, "dtype": dtype, "name": "get_session_tensor_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Another string handle, int32 dtype
    handle = np.array("tensor_handle_2", dtype=np.object_)
    dtype = np.int32
    input_dict = {"handle": handle, "dtype": dtype, "name": "get_session_tensor_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3:  string handle, complex64 dtype (Removed non-ascii characters)
    handle = np.array("tensor_handle_3", dtype=np.object_)
    dtype = np.complex64
    input_dict = {"handle": handle, "dtype": dtype, "name": "get_session_tensor_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty string handle, bool dtype
    handle = np.array("", dtype=np.object_)
    dtype = np.bool_
    input_dict = {"handle": handle, "dtype": dtype, "name": "get_session_tensor_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Long string handle, int64 dtype
    handle = np.array("a_very_long_tensor_handle_string_5", dtype=np.object_)
    dtype = np.int64
    input_dict = {"handle": handle, "dtype": dtype, "name": "get_session_tensor_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Handle with special characters, uint8 dtype
    handle = np.array("handle_with_$peci@l_chars", dtype=np.object_)
    dtype = np.uint8
    input_dict = {"handle": handle, "dtype": dtype, "name": "get_session_tensor_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Handle with numbers, float64 dtype
    handle = np.array("handle_1234567890", dtype=np.object_)
    dtype = np.float64
    input_dict = {"handle": handle, "dtype": dtype, "name": "get_session_tensor_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Handle with mixed characters, string dtype
    handle = np.array("MiXeD_cHaRs_HaNdLe", dtype=np.object_)
    dtype = np.string_
    input_dict = {"handle": handle, "dtype": dtype, "name": "get_session_tensor_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9:  Valid string handle, int8 dtype
    handle = np.array("handle_9", dtype=np.object_)
    dtype = np.int8
    input_dict = {"handle": handle, "dtype": dtype, "name": "get_session_tensor_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Valid string handle, uint16 dtype
    handle = np.array("handle_10", dtype=np.object_)
    dtype = np.uint16
    input_dict = {"handle": handle, "dtype": dtype, "name": "get_session_tensor_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.GetSessionTensor"] = tf_raw_ops_GetSessionTensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.GetSessionTensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.GetSessionTensor'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.GetSessionTensor', generated_inputs['tf.raw_ops.GetSessionTensor'], lib="tf", suffix=0)
