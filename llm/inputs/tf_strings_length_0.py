
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_length_inputs():
    list_of_inputs = []

    # Input 1: Basic string tensor, default unit
    input_tensor = tf.constant(["hello", "world"])
    unit_str = "BYTE"
    name_str = None
    input_dict = {"input": input_tensor, "unit": unit_str, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: String tensor with UTF8 chars, UTF8_CHAR unit
    input_tensor = tf.constant(["你好", "世界"])
    unit_str = "UTF8_CHAR"
    name_str = "length_utf8"
    input_dict = {"input": input_tensor, "unit": unit_str, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty string tensor
    input_tensor = tf.constant(["", ""])
    unit_str = "BYTE"
    name_str = None
    input_dict = {"input": input_tensor, "unit": unit_str, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Mixed UTF8 and ASCII
    input_tensor = tf.constant(["hello你好", "world世界"])
    unit_str = "UTF8_CHAR"
    name_str = None
    input_dict = {"input": input_tensor, "unit": unit_str, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with different length strings
    input_tensor = tf.constant(["a", "bb", "ccc"])
    unit_str = "BYTE"
    name_str = "different_lengths"
    input_dict = {"input": input_tensor, "unit": unit_str, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D Tensor
    input_tensor = tf.constant([["hello", "world"], ["你好", "世界"]])
    unit_str = "BYTE"
    name_str = None
    input_dict = {"input": input_tensor, "unit": unit_str, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with special characters
    input_tensor = tf.constant(["!@#$", "%^&*"])
    unit_str = "BYTE"
    name_str = None
    input_dict = {"input": input_tensor, "unit": unit_str, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with unicode emoji
    input_tensor = tf.constant(["\U0001F600", "\U0001F642"])
    unit_str = "UTF8_CHAR"
    name_str = "emoji_length"
    input_dict = {"input": input_tensor, "unit": unit_str, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: Tensor with long strings
    input_tensor = tf.constant(["This is a very long string", "Another very very long string"])
    unit_str = "BYTE"
    name_str = None
    input_dict = {"input": input_tensor, "unit": unit_str, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with numbers as strings
    input_tensor = tf.constant(["123", "45678"])
    unit_str = "BYTE"
    name_str = None
    input_dict = {"input": input_tensor, "unit": unit_str, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.length' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.length'.")

check_valid('tf.strings.length', generated_inputs['tf.strings.length'], lib="tf", suffix=0)
