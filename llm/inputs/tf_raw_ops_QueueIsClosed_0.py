
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QueueIsClosed_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.constant("test_queue")
    input_dict = {"handle": handle, "name": "queue_closed_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.constant("another_queue")
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.constant("queue_three")
    input_dict = {"handle": handle, "name": "queue_closed_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    handle = tf.constant("")
    input_dict = {"handle": handle, "name": "empty_queue"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.constant("some_queue")
    input_dict = {"handle": handle, "name": "named_queue"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.constant("yet_another_queue")
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    handle = tf.constant("queue_seven")
    input_dict = {"handle": handle, "name": "queue_closed_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.constant("8th_queue")
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.constant("ninth_queue")
    input_dict = {"handle": handle, "name": "queue_closed_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = tf.constant("tenth_queue")
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QueueIsClosed"] = tf_raw_ops_QueueIsClosed_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QueueIsClosed' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QueueIsClosed'.")

check_valid('tf.raw_ops.QueueIsClosed', generated_inputs['tf.raw_ops.QueueIsClosed'], lib="tf", suffix=0)
