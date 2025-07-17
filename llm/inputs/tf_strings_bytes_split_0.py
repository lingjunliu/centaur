
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_bytes_split_inputs():
    list_of_inputs = []

    # Input 1: Simple string tensor
    input_tensor = tf.constant("hello", dtype=tf.string).numpy()
    input_dict = {"input": input_tensor, "name": "split_string_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List of strings
    input_tensor = tf.constant(["hello", "world"], dtype=tf.string).numpy()
    input_dict = {"input": input_tensor, "name": "split_string_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty string
    input_tensor = tf.constant("", dtype=tf.string).numpy()
    input_dict = {"input": input_tensor, "name": "split_string_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: String with numbers and symbols
    input_tensor = tf.constant("123abc!@#", dtype=tf.string).numpy()
    input_dict = {"input": input_tensor, "name": "split_string_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: String with spaces
    input_tensor = tf.constant("hello world", dtype=tf.string).numpy()
    input_dict = {"input": input_tensor, "name": "split_string_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multidimensional tensor of strings
    input_tensor = tf.constant([["hello", "world"], ["foo", "bar"]], dtype=tf.string).numpy()
    input_dict = {"input": input_tensor, "name": "split_string_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with empty string
    input_tensor = tf.constant(["hello", ""], dtype=tf.string).numpy()
    input_dict = {"input": input_tensor, "name": "split_string_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with unicode characters (should split into bytes)
    input_tensor = tf.constant("你好世界", dtype=tf.string).numpy()
    input_dict = {"input": input_tensor, "name": "split_string_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with mixed characters
    input_tensor = tf.constant("hello你好", dtype=tf.string).numpy()
    input_dict = {"input": input_tensor, "name": "split_string_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: String with special byte sequences
    input_tensor = tf.constant(b'\x00\x01\xff'.decode('latin-1'), dtype=tf.string).numpy()
    input_dict = {"input": input_tensor, "name": "split_string_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strings.bytes_split"] = tf_strings_bytes_split_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.bytes_split' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.bytes_split'.")

check_valid('tf.strings.bytes_split', generated_inputs['tf.strings.bytes_split'], lib="tf", suffix=0)
