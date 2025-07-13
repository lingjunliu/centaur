
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_delete_session_tensor_inputs():
    list_of_inputs = []

    # Input 1
    handle = np.array(b"tensor_handle_1", dtype=np.string_)
    name = None
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = np.array(b"tensor_handle_2", dtype=np.string_)
    name = "delete_op_1"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = np.array(b"very_long_tensor_handle", dtype=np.string_)
    name = "complex_op_name"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = np.array(b"short", dtype=np.string_)
    name = ""
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = np.array(b"1234567890", dtype=np.string_)
    name = "op_5"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = np.array(b"handle_with_symbols!@#$", dtype=np.string_)
    name = "delete_with_symbols!@#$"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = np.array(b"handle_with_spaces", dtype=np.string_)
    name = "delete with spaces"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = np.array(b"another_handle", dtype=np.string_)
    name = None
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = np.array(b"", dtype=np.string_)
    name = "empty_handle"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = np.array(b"abcdefghijk", dtype=np.string_)
    name = "another_op"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
temp_list = tf_raw_ops_delete_session_tensor_inputs()
converted_list = []
for item in temp_list:
  converted_list.append({"kwargs": item})
generated_inputs["tf.raw_ops.DeleteSessionTensor"] = converted_list

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DeleteSessionTensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DeleteSessionTensor'.")

check_valid('tf.raw_ops.DeleteSessionTensor', generated_inputs['tf.raw_ops.DeleteSessionTensor'], lib="tf", suffix=0)
