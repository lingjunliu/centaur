
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ReaderReset_inputs():
    list_of_inputs = []

    # Input 1
    reader_handle = tf.constant("reader_handle_1", dtype=tf.string)
    name = "reset_op_1"
    input_dict = {"reader_handle": reader_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    reader_handle = tf.constant("reader_handle_2", dtype=tf.string)
    name = None
    input_dict = {"reader_handle": reader_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    reader_handle = tf.constant("reader_handle_3", dtype=tf.string)
    name = "reset_op_3"
    input_dict = {"reader_handle": reader_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    reader_handle = tf.constant("reader_handle_4", dtype=tf.string)
    name = None
    input_dict = {"reader_handle": reader_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    reader_handle = tf.constant("reader_handle_5", dtype=tf.string)
    name = "reset_op_5"
    input_dict = {"reader_handle": reader_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    reader_handle = tf.constant("reader_handle_6", dtype=tf.string)
    name = None
    input_dict = {"reader_handle": reader_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    reader_handle = tf.constant("reader_handle_7", dtype=tf.string)
    name = "reset_op_7"
    input_dict = {"reader_handle": reader_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    reader_handle = tf.constant("reader_handle_8", dtype=tf.string)
    name = None
    input_dict = {"reader_handle": reader_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    reader_handle = tf.constant("reader_handle_9", dtype=tf.string)
    name = "reset_op_9"
    input_dict = {"reader_handle": reader_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    reader_handle = tf.constant("reader_handle_10", dtype=tf.string)
    name = None
    input_dict = {"reader_handle": reader_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ReaderReset"] = tf_raw_ops_ReaderReset_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ReaderReset' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ReaderReset'.")

check_valid('tf.raw_ops.ReaderReset', generated_inputs['tf.raw_ops.ReaderReset'], lib="tf", suffix=0)
