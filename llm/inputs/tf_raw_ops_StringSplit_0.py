
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_string_split_inputs():
    list_of_inputs = []

    # Helper function to convert string arrays to bytes arrays
    def to_bytes(arr):
        return np.array([s.encode('utf-8') for s in arr]).astype(np.string_)

    # Input 1
    input_tensor = to_bytes(['hello world', 'a b c'])
    delimiter_tensor = to_bytes([' '])
    skip_empty = True
    name = None

    input_dict = {
        "input": input_tensor,
        "delimiter": delimiter_tensor,
        "skip_empty": skip_empty,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = to_bytes(['hello  world', 'a  b c'])
    delimiter_tensor = to_bytes([' '])
    skip_empty = False
    name = "test_split"

    input_dict = {
        "input": input_tensor,
        "delimiter": delimiter_tensor,
        "skip_empty": skip_empty,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = to_bytes(['hello,world', 'a,b,c'])
    delimiter_tensor = to_bytes([','])
    skip_empty = True
    name = None

    input_dict = {
        "input": input_tensor,
        "delimiter": delimiter_tensor,
        "skip_empty": skip_empty,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = to_bytes(['hello', 'world'])
    delimiter_tensor = to_bytes([''])
    skip_empty = True
    name = None

    input_dict = {
        "input": input_tensor,
        "delimiter": delimiter_tensor,
        "skip_empty": skip_empty,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    input_tensor = to_bytes(['', ''])
    delimiter_tensor = to_bytes([' '])
    skip_empty = True
    name = None

    input_dict = {
        "input": input_tensor,
        "delimiter": delimiter_tensor,
        "skip_empty": skip_empty,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = to_bytes(['abc'])
    delimiter_tensor = to_bytes(['bc'])
    skip_empty = True
    name = None

    input_dict = {
        "input": input_tensor,
        "delimiter": delimiter_tensor,
        "skip_empty": skip_empty,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = to_bytes(['a.b.c', 'd.e.f'])
    delimiter_tensor = to_bytes(['.'])
    skip_empty = False
    name = None

    input_dict = {
        "input": input_tensor,
        "delimiter": delimiter_tensor,
        "skip_empty": skip_empty,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = to_bytes(['hello world', ''])
    delimiter_tensor = to_bytes([' '])
    skip_empty = True
    name = None

    input_dict = {
        "input": input_tensor,
        "delimiter": delimiter_tensor,
        "skip_empty": skip_empty,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = to_bytes(['multiple delimiters', 'test||test'])
    delimiter_tensor = to_bytes(['| '])
    skip_empty = False
    name = None

    input_dict = {
        "input": input_tensor,
        "delimiter": delimiter_tensor,
        "skip_empty": skip_empty,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = to_bytes(['test'])
    delimiter_tensor = to_bytes(['test'])
    skip_empty = True
    name = None

    input_dict = {
        "input": input_tensor,
        "delimiter": delimiter_tensor,
        "skip_empty": skip_empty,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.StringSplit"] = tf_raw_ops_string_split_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.StringSplit' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StringSplit'.")

check_valid('tf.raw_ops.StringSplit', generated_inputs['tf.raw_ops.StringSplit'], lib="tf", suffix=0)
