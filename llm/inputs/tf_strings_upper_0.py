
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_upper_inputs():
    list_of_inputs = []

    # Input 1: Basic ASCII string
    input_tensor = tf.constant("hello world", dtype=tf.string)
    encoding = ""
    name = None
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: String with numbers and symbols
    input_tensor = tf.constant("123 abc!@#", dtype=tf.string)
    encoding = ""
    name = "numbers_symbols"
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty string
    input_tensor = tf.constant("", dtype=tf.string)
    encoding = ""
    name = None
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: String with uppercase already
    input_tensor = tf.constant("ALREADY UPPER", dtype=tf.string)
    encoding = ""
    name = "already_upper"
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: String with mixed case
    input_tensor = tf.constant("MiXeD CaSe", dtype=tf.string)
    encoding = ""
    name = None
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: UTF-8 string
    input_tensor = tf.constant("你好世界", dtype=tf.string)
    encoding = "utf-8"
    name = "utf8_string"
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: String with special characters
    input_tensor = tf.constant("string with \t tab and \n newline", dtype=tf.string)
    encoding = ""
    name = None
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Longer String
    input_tensor = tf.constant("a very long string that should still work correctly with uppercase conversion", dtype=tf.string)
    encoding = ""
    name = "long_string"
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: String with unicode characters and UTF-8 encoding
    input_tensor = tf.constant("café", dtype=tf.string)
    encoding = "utf-8"
    name = None
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: String with emojis (UTF-8)
    input_tensor = tf.constant("hello 😊 world", dtype=tf.string)
    encoding = "utf-8"
    name = "emoji_string"
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: 2D Tensor
    input_tensor = tf.constant([["hello", "world"], ["foo", "bar"]], dtype=tf.string)
    encoding = ""
    name = None
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strings.upper"] = tf_strings_upper_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.upper' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.upper'.")

check_valid('tf.strings.upper', generated_inputs['tf.strings.upper'], lib="tf", suffix=0)
