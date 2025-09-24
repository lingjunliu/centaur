
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_compat_as_str_any_inputs():
    list_of_inputs = []

    # Input 1, valid: Basic string
    value = "hello"
    encoding = "utf-8"
    input_dict = {"value": value, "encoding": encoding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid: Empty string
    value = ""
    encoding = "utf-8"
    input_dict = {"value": value, "encoding": encoding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid: String with special characters
    value = "!@#$%^&*()"
    encoding = "utf-8"
    input_dict = {"value": value, "encoding": encoding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid: String with numbers
    value = "1234567890"
    encoding = "utf-8"
    input_dict = {"value": value, "encoding": encoding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid: String with mixed characters
    value = "a1b2c3d4e5"
    encoding = "utf-8"
    input_dict = {"value": value, "encoding": encoding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid: String with spaces
    value = "hello world"
    encoding = "utf-8"
    input_dict = {"value": value, "encoding": encoding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid: Long string
    value = "This is a very long string to test the function."
    encoding = "utf-8"
    input_dict = {"value": value, "encoding": encoding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid: Different encoding
    value = "hello"
    encoding = "ascii"
    input_dict = {"value": value, "encoding": encoding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid: Unicode string
    value = "你好世界"
    encoding = "utf-8"
    input_dict = {"value": value, "encoding": encoding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid: String with newline
    value = "hello\nworld"
    encoding = "utf-8"
    input_dict = {"value": value, "encoding": encoding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.compat.as_str_any"] = tf_compat_as_str_any_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.compat.as_str_any' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.compat.as_str_any'.")

check_valid('tf.compat.as_str_any', generated_inputs['tf.compat.as_str_any'], lib="tf", suffix=0)
