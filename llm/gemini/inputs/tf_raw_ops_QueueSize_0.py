
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_QueueSize_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.constant("test_queue_1", dtype=tf.string).numpy().decode('utf-8')
    input_dict = {"handle": handle, "name": "queue_size_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.constant("test_queue_2", dtype=tf.string).numpy().decode('utf-8')
    input_dict = {"handle": handle, "name": "queue_size_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.constant("test_queue_3", dtype=tf.string).numpy().decode('utf-8')
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.constant("test_queue_4", dtype=tf.string).numpy().decode('utf-8')
    input_dict = {"handle": handle, "name": "queue_size_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.constant("test_queue_5", dtype=tf.string).numpy().decode('utf-8')
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.constant("long_queue_name_6", dtype=tf.string).numpy().decode('utf-8')
    input_dict = {"handle": handle, "name": "queue_size_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = tf.constant("test_queue_7", dtype=tf.string).numpy().decode('utf-8')
    input_dict = {"handle": handle, "name": "very_long_name_for_queue_size_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8
    handle = tf.constant("test_queue_8", dtype=tf.string).numpy().decode('utf-8')
    input_dict = {"handle": handle, "name": "queue_size_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.constant("test_queue_9", dtype=tf.string).numpy().decode('utf-8')
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = tf.constant("another_test_queue_10", dtype=tf.string).numpy().decode('utf-8')
    input_dict = {"handle": handle, "name": "queue_size_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QueueSize"] = tf_raw_ops_QueueSize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QueueSize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QueueSize'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.QueueSize', generated_inputs['tf.raw_ops.QueueSize'], lib="tf", suffix=0)
