
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

def tf_raw_ops_delete_session_tensor_inputs():
    list_of_inputs = []

    # Input 1
    handle = np.array(b"tensor_handle_1", dtype=np.string_)
    name = None
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(input_dict)

    # Input 2
    handle = np.array(b"another_handle", dtype=np.string_)
    name = "delete_op_1"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(input_dict)

    # Input 3
    handle = np.array(b"yet_another_handle_123", dtype=np.string_)
    name = "delete_op_2"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(input_dict)

    # Input 4
    handle = np.array(b"", dtype=np.string_)
    name = "delete_op_3"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(input_dict)

    # Input 5
    handle = np.array(b"handle_with_numbers_12345", dtype=np.string_)
    name = None
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(input_dict)

    # Input 6
    handle = np.array(b"handle_with_special_chars!@#$", dtype=np.string_)
    name = "delete_op_4"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(input_dict)

    # Input 7
    handle = np.array(b"a_very_long_handle_" + b"a"*50, dtype=np.string_)
    name = None
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(input_dict)

    # Input 8
    handle = np.array(b"handle_with_unicode", dtype=np.string_)
    name = "delete_op_5"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(input_dict)

    # Input 9
    handle = np.array(b"handle_with_newline\ncharacter", dtype=np.string_)
    name = None
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(input_dict)

    # Input 10
    handle = np.array(b"handle_with_tab\tcharacter", dtype=np.string_)
    name = "delete_op_6"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
api_name = "tf.raw_ops.DeleteSessionTensor"
inputs = tf_raw_ops_delete_session_tensor_inputs()
generated_inputs[api_name] = []
for input_dict in inputs:
  generated_inputs[api_name].append({"kwargs": {"handle": input_dict["handle"], "name": input_dict["name"]}})

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DeleteSessionTensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DeleteSessionTensor'.")

check_valid('tf.raw_ops.DeleteSessionTensor', generated_inputs['tf.raw_ops.DeleteSessionTensor'], lib="tf", suffix=0)
