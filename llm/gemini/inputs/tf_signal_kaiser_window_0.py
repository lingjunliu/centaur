
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_kaiser_window_inputs():
    list_of_inputs = []

    # Input 1
    window_length = tf.constant(10, dtype=tf.int32).numpy()
    beta = 12.0
    dtype = tf.float32
    name = "kaiser_window_1"
    input_dict = {"window_length": window_length, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    window_length = tf.constant(20, dtype=tf.int64).numpy()
    beta = 5.0
    dtype = tf.float64
    name = "kaiser_window_2"
    input_dict = {"window_length": window_length, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    window_length = tf.constant(5, dtype=tf.int32).numpy()
    beta = 0.0
    dtype = tf.float32
    name = "kaiser_window_3"
    input_dict = {"window_length": window_length, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    window_length = tf.constant(15, dtype=tf.int32).numpy()
    beta = 8.5
    dtype = tf.float64
    name = "kaiser_window_4"
    input_dict = {"window_length": window_length, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    window_length = tf.constant(30, dtype=tf.int64).numpy()
    beta = 15.0
    dtype = tf.float32
    name = "kaiser_window_5"
    input_dict = {"window_length": window_length, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    window_length = tf.constant(7, dtype=tf.int32).numpy()
    beta = 2.0
    dtype = tf.float64
    name = "kaiser_window_6"
    input_dict = {"window_length": window_length, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    window_length = tf.constant(25, dtype=tf.int64).numpy()
    beta = 7.2
    dtype = tf.float32
    name = "kaiser_window_7"
    input_dict = {"window_length": window_length, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    window_length = tf.constant(12, dtype=tf.int32).numpy()
    beta = 10.0
    dtype = tf.float64
    name = "kaiser_window_8"
    input_dict = {"window_length": window_length, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9
    window_length = tf.constant(1, dtype=tf.int32).numpy()
    beta = 1.0
    dtype = tf.float32
    name = "kaiser_window_9"
    input_dict = {"window_length": window_length, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    window_length = tf.constant(64, dtype=tf.int32).numpy()
    beta = 64.0
    dtype = tf.float64
    name = "kaiser_window_10"
    input_dict = {"window_length": window_length, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.signal.kaiser_window"] = tf_signal_kaiser_window_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.kaiser_window' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.kaiser_window'.")

check_valid('tf.signal.kaiser_window', generated_inputs['tf.signal.kaiser_window'], lib="tf", suffix=0)
