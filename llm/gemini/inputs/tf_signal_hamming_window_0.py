
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_hamming_window_inputs():
    list_of_inputs = []

    # Input 1
    window_length = tf.constant(5, dtype=tf.int32).numpy()
    periodic = True
    dtype = tf.float32
    name = "hamming_window_1"

    input_dict = {
        "window_length": window_length,
        "periodic": periodic,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    window_length = tf.constant(10, dtype=tf.int32).numpy()
    periodic = False
    dtype = tf.float64
    name = "hamming_window_2"

    input_dict = {
        "window_length": window_length,
        "periodic": periodic,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    window_length = tf.constant(1, dtype=tf.int32).numpy()
    periodic = True
    dtype = tf.float32
    name = "hamming_window_3"

    input_dict = {
        "window_length": window_length,
        "periodic": periodic,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    window_length = tf.constant(20, dtype=tf.int32).numpy()
    periodic = False
    dtype = tf.float64
    name = "hamming_window_4"

    input_dict = {
        "window_length": window_length,
        "periodic": periodic,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    window_length = tf.constant(7, dtype=tf.int32).numpy()
    periodic = True
    dtype = tf.float16
    name = "hamming_window_5"

    input_dict = {
        "window_length": window_length,
        "periodic": periodic,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    window_length = tf.constant(3, dtype=tf.int32).numpy()
    periodic = False
    dtype = tf.float16
    name = "hamming_window_6"

    input_dict = {
        "window_length": window_length,
        "periodic": periodic,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    window_length = tf.constant(15, dtype=tf.int32).numpy()
    periodic = True
    dtype = tf.float32
    name = "hamming_window_7"

    input_dict = {
        "window_length": window_length,
        "periodic": periodic,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    window_length = tf.constant(25, dtype=tf.int32).numpy()
    periodic = False
    dtype = tf.float64
    name = "hamming_window_8"

    input_dict = {
        "window_length": window_length,
        "periodic": periodic,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    window_length = tf.constant(30, dtype=tf.int32).numpy()
    periodic = True
    dtype = tf.float16
    name = "hamming_window_9"

    input_dict = {
        "window_length": window_length,
        "periodic": periodic,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    window_length = tf.constant(12, dtype=tf.int32).numpy()
    periodic = False
    dtype = tf.float32
    name = "hamming_window_10"

    input_dict = {
        "window_length": window_length,
        "periodic": periodic,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.signal.hamming_window"] = tf_signal_hamming_window_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.hamming_window' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.hamming_window'.")

check_valid('tf.signal.hamming_window', generated_inputs['tf.signal.hamming_window'], lib="tf", suffix=0)
