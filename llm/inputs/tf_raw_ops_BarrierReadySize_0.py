
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_BarrierReadySize_inputs():
    list_of_inputs = []

    # Input 1: Valid handle
    handle_1 = tf.constant("barrier_handle_1", dtype=tf.string).numpy()
    input_dict_1 = {"handle": handle_1, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Another valid handle
    handle_2 = tf.constant("another_barrier_handle", dtype=tf.string).numpy()
    input_dict_2 = {"handle": handle_2, "name": "my_barrier"}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Handle with a different name
    handle_3 = tf.constant("barrier_ready_size_test", dtype=tf.string).numpy()
    input_dict_3 = {"handle": handle_3, "name": "ready_size"}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Empty string handle
    handle_4 = tf.constant("", dtype=tf.string).numpy()
    input_dict_4 = {"handle": handle_4, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Handle with unicode string
    handle_5 = tf.constant("你好", dtype=tf.string).numpy()
    input_dict_5 = {"handle": handle_5, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Handle with special characters
    handle_6 = tf.constant("!@#$%^&*()", dtype=tf.string).numpy()
    input_dict_6 = {"handle": handle_6, "name": "special_chars"}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Handle with numbers
    handle_7 = tf.constant("12345", dtype=tf.string).numpy()
    input_dict_7 = {"handle": handle_7, "name": "numbers"}
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: Handle with combination of chars and numbers
    handle_8 = tf.constant("abc123xyz", dtype=tf.string).numpy()
    input_dict_8 = {"handle": handle_8, "name": "alphanumeric"}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Long handle name
    handle_9 = tf.constant("very_long_handle_name_to_test_max_length", dtype=tf.string).numpy()
    input_dict_9 = {"handle": handle_9, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Handle with spaces
    handle_10 = tf.constant("handle with spaces", dtype=tf.string).numpy()
    input_dict_10 = {"handle": handle_10, "name": "spaced"}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.BarrierReadySize"] = tf_raw_ops_BarrierReadySize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.BarrierReadySize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BarrierReadySize'.")

check_valid('tf.raw_ops.BarrierReadySize', generated_inputs['tf.raw_ops.BarrierReadySize'], lib="tf", suffix=0)
