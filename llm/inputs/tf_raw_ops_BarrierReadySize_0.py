
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_barrier_ready_size_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.constant(np.array(b"barrier_handle_1"), dtype=tf.string)
    name = "op_name_1"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.constant(np.array(b"barrier_handle_2"), dtype=tf.string)
    name = None
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.constant(np.array(b"another_barrier"), dtype=tf.string)
    name = "ready_size_op"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.constant(np.array(b""), dtype=tf.string)
    name = "empty_handle"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.constant(np.array(b"barrier_42"), dtype=tf.string)
    name = ""
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.constant(np.array(b"very_long_barrier_handle_name"), dtype=tf.string)
    name = "a_very_long_op_name"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    handle = tf.constant(np.array(b"handle_with_numbers_123"), dtype=tf.string)
    name = "numbers_in_name_456"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.constant(np.array(b"handle_with_special_chars"), dtype=tf.string)
    name = "special_chars"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.constant(np.array(b"UTF-8_handle"), dtype=tf.string)
    name = "UTF-8_name"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = tf.constant(np.array(b"handle_with_space"), dtype=tf.string)
    name = "name with space"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.BarrierReadySize"] = tf_raw_ops_barrier_ready_size_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.BarrierReadySize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BarrierReadySize'.")

check_valid('tf.raw_ops.BarrierReadySize', generated_inputs['tf.raw_ops.BarrierReadySize'], lib="tf", suffix=0)
