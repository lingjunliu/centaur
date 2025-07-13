
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_string_join_inputs():
    list_of_inputs = []

    # Input 1: Basic example with scalar separator
    inputs = [tf.constant("hello"), tf.constant("world"), tf.constant("tensorflow")]
    separator = " "
    name = None
    input_dict = {"inputs": inputs, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Empty separator
    inputs = [tf.constant("hello"), tf.constant("world")]
    separator = ""
    name = "join_no_sep"
    input_dict = {"inputs": inputs, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tensor separator
    inputs = [tf.constant(["a", "b"]), tf.constant(["c", "d"])]
    separator = "-"
    name = None
    input_dict = {"inputs": inputs, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar inputs
    inputs = [tf.constant("one"), tf.constant("two"), tf.constant("three")]
    separator = ","
    name = "scalar_join"
    input_dict = {"inputs": inputs, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Mixed scalars and tensors
    inputs = [tf.constant("prefix"), tf.constant(["a", "b", "c"])]
    separator = "_"
    name = None
    input_dict = {"inputs": inputs, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Longer separator
    inputs = [tf.constant("part1"), tf.constant("part2"), tf.constant("part3")]
    separator = "---separator---"
    name = None
    input_dict = {"inputs": inputs, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: inputs of differing length but still valid after broadcasting
    inputs = [tf.constant(["hello", "world"]), tf.constant("tensorflow")]
    separator = " "
    name = None
    input_dict = {"inputs": inputs, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Unicode strings
    inputs = [tf.constant("你好"), tf.constant("世界")]
    separator = " "
    name = "unicode_join"
    input_dict = {"inputs": inputs, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty tensors
    inputs = [tf.constant(""), tf.constant("")]
    separator = " "
    name = None
    input_dict = {"inputs": inputs, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single input
    inputs = [tf.constant("single")]
    separator = " "
    name = None
    input_dict = {"inputs": inputs, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.StringJoin"] = tf_raw_ops_string_join_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.StringJoin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StringJoin'.")

check_valid('tf.raw_ops.StringJoin', generated_inputs['tf.raw_ops.StringJoin'], lib="tf", suffix=0)
