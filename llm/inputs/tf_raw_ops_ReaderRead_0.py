
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ReaderRead_inputs():
    list_of_inputs = []

    # Input 1
    reader_handle = tf.compat.v1.get_variable("reader_handle_1", shape=[], dtype=tf.string, initializer=tf.compat.v1.zeros_initializer())
    queue_handle = tf.compat.v1.get_variable("queue_handle_1", shape=[], dtype=tf.string, initializer=tf.compat.v1.zeros_initializer())
    name = "reader_read_op_1"
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    reader_handle = tf.compat.v1.get_variable("reader_handle_2", shape=[], dtype=tf.string, initializer=tf.compat.v1.zeros_initializer())
    queue_handle = tf.compat.v1.get_variable("queue_handle_2", shape=[], dtype=tf.string, initializer=tf.compat.v1.zeros_initializer())
    name = None
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    reader_handle = tf.compat.v1.get_variable("reader_handle_3", shape=[], dtype=tf.string, initializer=tf.compat.v1.zeros_initializer())
    queue_handle = tf.compat.v1.get_variable("queue_handle_3", shape=[], dtype=tf.string, initializer=tf.compat.v1.zeros_initializer())
    name = "another_reader_op"
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    reader_handle = tf.compat.v1.get_variable("reader_handle_4", shape=[], dtype=tf.string, initializer=tf.compat.v1.zeros_initializer())
    queue_handle = tf.compat.v1.get_variable("queue_handle_4", shape=[], dtype=tf.string, initializer=tf.compat.v1.zeros_initializer())
    name = ""
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    reader_handle = tf.compat.v1.get_variable("reader_handle_5", shape=[], dtype=tf.string, initializer=tf.compat.v1.zeros_initializer())
    queue_handle = tf.compat.v1.get_variable("queue_handle_5", shape=[], dtype=tf.string, initializer=tf.compat.v1.zeros_initializer())
    name = "a_very_long_name_for_an_operation"
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    reader_handle = tf.compat.v1.get_variable("reader_6", shape=[], dtype=tf.string, initializer=tf.compat.v1.zeros_initializer())
    queue_handle = tf.compat.v1.get_variable("queue_6", shape=[], dtype=tf.string, initializer=tf.compat.v1.zeros_initializer())
    name = "reader_read_6"
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    reader_handle = tf.compat.v1.get_variable("reader_7", shape=[], dtype=tf.string, initializer=tf.compat.v1.zeros_initializer())
    queue_handle = tf.compat.v1.get_variable("queue_7", shape=[], dtype=tf.string, initializer=tf.compat.v1.zeros_initializer())
    name = "reader_read_7"
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    reader_handle = tf.compat.v1.get_variable("reader_8", shape=[], dtype=tf.string, initializer=tf.compat.v1.zeros_initializer())
    queue_handle = tf.compat.v1.get_variable("queue_8", shape=[], dtype=tf.string, initializer=tf.compat.v1.zeros_initializer())
    name = "reader_read_8"
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    reader_handle = tf.compat.v1.get_variable("reader_9", shape=[], dtype=tf.string, initializer=tf.compat.v1.zeros_initializer())
    queue_handle = tf.compat.v1.get_variable("queue_9", shape=[], dtype=tf.string, initializer=tf.compat.v1.zeros_initializer())
    name = "reader_read_9"
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    reader_handle = tf.compat.v1.get_variable("reader_10", shape=[], dtype=tf.string, initializer=tf.compat.v1.zeros_initializer())
    queue_handle = tf.compat.v1.get_variable("queue_handle_10", shape=[], dtype=tf.string, initializer=tf.compat.v1.zeros_initializer())
    name = "reader_read_10"
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ReaderRead"] = tf_raw_ops_ReaderRead_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ReaderRead' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ReaderRead'.")

check_valid('tf.raw_ops.ReaderRead', generated_inputs['tf.raw_ops.ReaderRead'], lib="tf", suffix=0)
