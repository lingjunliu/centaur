
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_encode_base64_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array(["hello"]).astype(np.string_)
    pad_bool = False
    name_str = None
    input_dict = {"input": input_tensor, "pad": pad_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array(["hello", "world"]).astype(np.string_)
    pad_bool = True
    name_str = "encode_1"
    input_dict = {"input": input_tensor, "pad": pad_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([""]).astype(np.string_)
    pad_bool = False
    name_str = "encode_2"
    input_dict = {"input": input_tensor, "pad": pad_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array(["a", "b", "c"]).astype(np.string_)
    pad_bool = True
    name_str = None
    input_dict = {"input": input_tensor, "pad": pad_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array(["This is a longer string."]).astype(np.string_)
    pad_bool = False
    name_str = "encode_3"
    input_dict = {"input": input_tensor, "pad": pad_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([["hello", "world"], ["foo", "bar"]]).astype(np.string_)
    pad_bool = True
    name_str = None
    input_dict = {"input": input_tensor, "pad": pad_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array(["12345", "67890"]).astype(np.string_)
    pad_bool = False
    name_str = "encode_4"
    input_dict = {"input": input_tensor, "pad": pad_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array(["!@#$%^", "&*()_+"]).astype(np.string_)
    pad_bool = True
    name_str = None
    input_dict = {"input": input_tensor, "pad": pad_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array(["test_string"]).astype(np.string_)
    pad_bool = False
    name_str = "encode_5"
    input_dict = {"input": input_tensor, "pad": pad_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([["a"], ["b"]]).astype(np.string_)
    pad_bool = True
    name_str = None
    input_dict = {"input": input_tensor, "pad": pad_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.EncodeBase64"] = tf_raw_ops_encode_base64_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.EncodeBase64' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.EncodeBase64'.")

check_valid('tf.raw_ops.EncodeBase64', generated_inputs['tf.raw_ops.EncodeBase64'], lib="tf", suffix=0)
