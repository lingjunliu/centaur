
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_stringjoin_inputs():
    list_of_inputs = []

    # Input 1: Basic test with separator
    inputs = [tf.constant("hello"), tf.constant("world")]
    separator = " "
    name = None
    input_dict = {"inputs": inputs, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: No separator
    inputs = [tf.constant("foo"), tf.constant("bar")]
    separator = ""
    name = "string_join_no_sep"
    input_dict = {"inputs": inputs, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple strings
    inputs = [tf.constant("this"), tf.constant("is"), tf.constant("a"), tf.constant("test")]
    separator = "-"
    name = None
    input_dict = {"inputs": inputs, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty strings
    inputs = [tf.constant(""), tf.constant(""), tf.constant("")]
    separator = ","
    name = "string_join_empty"
    input_dict = {"inputs": inputs, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Scalar and non-scalar - fixed shape
    inputs = [tf.constant(["a", "b"]), tf.constant(["c", "d"])]
    separator = "|"
    name = None
    input_dict = {"inputs": inputs, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Scalar input only
    inputs = [tf.constant("single")]
    separator = "-"
    name = None
    input_dict = {"inputs": inputs, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: More complex separator
    inputs = [tf.constant("one"), tf.constant("two"), tf.constant("three")]
    separator = ":::"
    name = None
    input_dict = {"inputs": inputs, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Unicode string input
    inputs = [tf.constant("你好"), tf.constant("世界")]
    separator = ""
    name = "unicode_string"
    input_dict = {"inputs": inputs, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensors with more dimensions
    inputs = [tf.constant([["a", "b"], ["c", "d"]]), tf.constant([["e", "f"], ["g", "h"]])]
    separator = ","
    name = None
    input_dict = {"inputs": inputs, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty separator
    inputs = [tf.constant("test"), tf.constant("test2")]
    separator = ""
    name = None
    input_dict = {"inputs": inputs, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.StringJoin"] = tf_raw_ops_stringjoin_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.StringJoin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StringJoin'.")

check_valid('tf.raw_ops.StringJoin', generated_inputs['tf.raw_ops.StringJoin'], lib="tf", suffix=0)
