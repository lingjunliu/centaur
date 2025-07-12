
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_AccumulatorNumAccumulated_inputs():
    list_of_inputs = []

    # Input 1, valid
    handle = tf.Variable("accumulator_handle_1", dtype=tf.string)
    name = "AccumulatorNumAccumulated_1"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    handle = tf.Variable("accumulator_handle_2", dtype=tf.string)
    name = None
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    handle = tf.Variable("another_accumulator", dtype=tf.string)
    name = "AnotherName"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    handle = tf.Variable("yet_another_handle", dtype=tf.string)
    name = ""
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid - different handle name
    handle = tf.Variable("diff_handle", dtype=tf.string)
    name = "DiffName"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid - longer name
    handle = tf.Variable("long_handle_name", dtype=tf.string)
    name = "Averylonganddescriptiveopname"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid
    handle = tf.Variable("handle7", dtype=tf.string)
    name = "seven"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid
    handle = tf.Variable("handle_number_8", dtype=tf.string)
    name = "eight8"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid
    handle = tf.Variable("handle9", dtype=tf.string)
    name = "name9"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid
    handle = tf.Variable("tenth_handle", dtype=tf.string)
    name = "tenth_name"
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
