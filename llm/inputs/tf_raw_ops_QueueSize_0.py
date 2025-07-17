
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_queue_size_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.constant("test_queue_1", dtype=tf.string)
    input_dict = {"handle": handle}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.constant("test_queue_2", dtype=tf.string)
    input_dict = {"handle": handle}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.constant("test_queue_3", dtype=tf.string)
    input_dict = {"handle": handle}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.constant("queue_4", dtype=tf.string)
    input_dict = {"handle": handle}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.constant("queue_5", dtype=tf.string)
    input_dict = {"handle": handle}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.constant("very_long_queue_name", dtype=tf.string)
    input_dict = {"handle": handle}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = tf.constant("queue_7_with_numbers_123", dtype=tf.string)
    input_dict = {"handle": handle}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.constant("queue_8_with_special_chars!@#$", dtype=tf.string)
    input_dict = {"handle": handle}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    handle = tf.constant("", dtype=tf.string)
    input_dict = {"handle": handle}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    handle = tf.constant("queue_10", dtype=tf.string)
    input_dict = {"handle": handle}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QueueSize"] = tf_raw_ops_queue_size_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QueueSize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QueueSize'.")

check_valid('tf.raw_ops.QueueSize', generated_inputs['tf.raw_ops.QueueSize'], lib="tf", suffix=0)
