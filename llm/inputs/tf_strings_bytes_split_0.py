
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_bytes_split_inputs():
    list_of_inputs = []

    # Input 1: Simple string tensor
    input_tensor = tf.constant("hello", dtype=tf.string)
    name = None
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List of strings
    input_tensor = tf.constant(["hello", "world"], dtype=tf.string)
    name = "split_strings"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty string
    input_tensor = tf.constant("", dtype=tf.string)
    name = None
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with a single empty string
    input_tensor = tf.constant([""], dtype=tf.string)
    name = None
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: String with numbers and special characters
    input_tensor = tf.constant("123!@#", dtype=tf.string)
    name = "special_chars"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensor of strings with varying lengths
    input_tensor = tf.constant(["a", "bb", "ccc"], dtype=tf.string)
    name = None
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor of strings with unicode characters
    input_tensor = tf.constant(["你好", "世界"], dtype=tf.string)
    name = "unicode"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multidimensional tensor
    input_tensor = tf.constant([["hello", "world"], ["foo", "bar"]], dtype=tf.string)
    name = "multi_dim"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor containing both empty and non-empty strings
    input_tensor = tf.constant(["hello", "", "world"], dtype=tf.string)
    name = None
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Simple byte string
    input_tensor = tf.constant(b"bytes")
    name = None
    input_dict = {"input": input_tensor, "name": name}
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
