
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_BarrierIncompleteSize_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.constant("barrier_0", dtype=tf.string)
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.constant("barrier_1", dtype=tf.string)
    input_dict = {"handle": handle, "name": b"my_barrier"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.constant("barrier_2", dtype=tf.string)
    input_dict = {"handle": handle, "name": b""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.constant("another_barrier", dtype=tf.string)
    input_dict = {"handle": handle, "name": b"another_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.constant("barrier_4", dtype=tf.string)
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    handle = tf.constant("barrier_5", dtype=tf.string)
    input_dict = {"handle": handle, "name": b"barrier_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = tf.constant("barrier_6", dtype=tf.string)
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.constant("barrier_7", dtype=tf.string)
    input_dict = {"handle": handle, "name": b"different_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.constant("barrier_8", dtype=tf.string)
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = tf.constant("barrier_9", dtype=tf.string)
    input_dict = {"handle": handle, "name": b"last_barrier"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.BarrierIncompleteSize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BarrierIncompleteSize'.")

check_valid('tf.raw_ops.BarrierIncompleteSize', generated_inputs['tf.raw_ops.BarrierIncompleteSize'], lib="tf", suffix=0)
