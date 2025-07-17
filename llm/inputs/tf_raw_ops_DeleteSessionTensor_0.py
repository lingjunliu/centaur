
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DeleteSessionTensor_inputs():
    list_of_inputs = []

    # Input 1: Simple handle
    handle = "tensor_handle_1"
    name = None
    input_dict = {"handle": handle.encode('utf-8'), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Another simple handle
    handle = "tensor_handle_2"
    name = "delete_op_1"
    input_dict = {"handle": handle.encode('utf-8'), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Handle with special characters
    handle = "tensor.handle-3_"
    name = None
    input_dict = {"handle": handle.encode('utf-8'), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Longer handle
    handle = "very_long_tensor_handle_4_with_underscores"
    name = "delete_op_2"
    input_dict = {"handle": handle.encode('utf-8'), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Handle with numbers
    handle = "tensor_handle_5_123"
    name = None
    input_dict = {"handle": handle.encode('utf-8'), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Handle with mixed characters
    handle = "tensor.Handle-6_123"
    name = "delete_op_3"
    input_dict = {"handle": handle.encode('utf-8'), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: Short handle
    handle = "a"
    name = None
    input_dict = {"handle": handle.encode('utf-8'), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Longer name
    handle = "tensor_handle_8"
    name = "very_long_delete_op_name_8"
    input_dict = {"handle": handle.encode('utf-8'), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty name
    handle = "tensor_handle_9"
    name = ""
    input_dict = {"handle": handle.encode('utf-8'), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty handle
    handle = ""
    name = None
    input_dict = {"handle": handle.encode('utf-8'), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DeleteSessionTensor"] = tf_raw_ops_DeleteSessionTensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DeleteSessionTensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DeleteSessionTensor'.")

check_valid('tf.raw_ops.DeleteSessionTensor', generated_inputs['tf.raw_ops.DeleteSessionTensor'], lib="tf", suffix=0)
