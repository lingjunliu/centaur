
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_AccumulatorNumAccumulated_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.constant("accumulator_handle_1")
    name = "AccumulatorNumAccumulated_1"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.constant("accumulator_handle_2")
    name = None
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.constant("accumulator_handle_3")
    name = "AccumulatorNumAccumulated_3"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.constant("accumulator_handle_4")
    name = None
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.constant("accumulator_handle_5")
    name = "AccumulatorNumAccumulated_5"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.constant("another_handle")
    name = None
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    handle = tf.constant("handle_7")
    name = "AccumulatorNumAccumulated_7"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.constant("handle_8")
    name = None
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.constant("handle_9")
    name = "AccumulatorNumAccumulated_9"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = tf.constant("handle_10")
    name = None
    input_dict = {"handle": handle, "name": name}
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
