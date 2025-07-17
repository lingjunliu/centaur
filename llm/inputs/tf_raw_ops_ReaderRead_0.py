
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ReaderRead_inputs():
    list_of_inputs = []

    # Input 1
    reader_handle = tf.constant("reader_handle", dtype=tf.string)
    queue_handle = tf.constant("queue_handle", dtype=tf.string)
    name = "test_read_1"
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    reader_handle = tf.constant("another_reader", dtype=tf.string)
    queue_handle = tf.constant("another_queue", dtype=tf.string)
    name = None
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    reader_handle = tf.constant("reader3", dtype=tf.string)
    queue_handle = tf.constant("queue3", dtype=tf.string)
    name = "read3"
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    reader_handle = tf.constant("reader4", dtype=tf.string)
    queue_handle = tf.constant("queue4", dtype=tf.string)
    name = ""
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    reader_handle = tf.constant("reader5", dtype=tf.string)
    queue_handle = tf.constant("queue5", dtype=tf.string)
    name = "test_read_5"
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    reader_handle = tf.constant("reader6", dtype=tf.string)
    queue_handle = tf.constant("queue6", dtype=tf.string)
    name = "read6"
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    reader_handle = tf.constant("reader7", dtype=tf.string)
    queue_handle = tf.constant("queue7", dtype=tf.string)
    name = None
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    reader_handle = tf.constant("reader8", dtype=tf.string)
    queue_handle = tf.constant("queue8", dtype=tf.string)
    name = "test_reader_8"
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    reader_handle = tf.constant("reader9", dtype=tf.string)
    queue_handle = tf.constant("queue9", dtype=tf.string)
    name = "reader_read_9"
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    reader_handle = tf.constant("reader10", dtype=tf.string)
    queue_handle = tf.constant("queue10", dtype=tf.string)
    name = "reader_read_test_10"
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
