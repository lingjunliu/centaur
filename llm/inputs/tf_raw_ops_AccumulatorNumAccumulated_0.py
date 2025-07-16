
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_AccumulatorNumAccumulated_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.compat.as_bytes("accumulator_handle_1")
    handle = tf.convert_to_tensor(handle, dtype=tf.string)
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.compat.as_bytes("accumulator_handle_2")
    handle = tf.convert_to_tensor(handle, dtype=tf.string)
    input_dict = {"handle": handle, "name": "AccumulatorCount"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.compat.as_bytes("long_accumulator_handle")
    handle = tf.convert_to_tensor(handle, dtype=tf.string)
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.compat.as_bytes("another_handle_example")
    handle = tf.convert_to_tensor(handle, dtype=tf.string)
    input_dict = {"handle": handle, "name": "DifferentName"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.compat.as_bytes("handle_5")
    handle = tf.convert_to_tensor(handle, dtype=tf.string)
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.compat.as_bytes("")
    handle = tf.convert_to_tensor(handle, dtype=tf.string)
    input_dict = {"handle": handle, "name": "EmptyHandle"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = tf.compat.as_bytes(" ")
    handle = tf.convert_to_tensor(handle, dtype=tf.string)
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.compat.as_bytes("handle_with_numbers_123")
    handle = tf.convert_to_tensor(handle, dtype=tf.string)
    input_dict = {"handle": handle, "name": "HandleWithNumbers"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.compat.as_bytes("handle_with_symbols_!@#$")
    handle = tf.convert_to_tensor(handle, dtype=tf.string)
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    handle = tf.compat.as_bytes("unicode_handle_你好世界")
    handle = tf.convert_to_tensor(handle, dtype=tf.string)
    input_dict = {"handle": handle, "name": "UnicodeHandle"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AccumulatorNumAccumulated"] = tf_raw_ops_AccumulatorNumAccumulated_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.AccumulatorNumAccumulated' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AccumulatorNumAccumulated'.")

check_valid('tf.raw_ops.AccumulatorNumAccumulated', generated_inputs['tf.raw_ops.AccumulatorNumAccumulated'], lib="tf", suffix=0)
