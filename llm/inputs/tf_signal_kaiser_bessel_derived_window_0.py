
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_kaiser_bessel_derived_window_inputs():
    list_of_inputs = []

    # Input 1, valid
    window_length = tf.constant(10, dtype=tf.int32).numpy()
    beta = 12.0
    dtype = tf.float32
    name = "window1"
    input_dict = {"window_length": window_length, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    window_length = tf.constant(256, dtype=tf.int32).numpy()
    beta = 5.0
    dtype = tf.float64
    name = "window2"
    input_dict = {"window_length": window_length, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    window_length = tf.constant(1, dtype=tf.int32).numpy()
    beta = 0.0
    dtype = tf.float32
    name = "window3"
    input_dict = {"window_length": window_length, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    window_length = tf.constant(512, dtype=tf.int32).numpy()
    beta = 15.0
    dtype = tf.float64
    name = "window4"
    input_dict = {"window_length": window_length, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    window_length = tf.constant(32, dtype=tf.int32).numpy()
    beta = 8.0
    dtype = tf.float32
    name = "window5"
    input_dict = {"window_length": window_length, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid
    window_length = tf.constant(64, dtype=tf.int32).numpy()
    beta = 2.0
    dtype = tf.float64
    name = "window6"
    input_dict = {"window_length": window_length, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid
    window_length = tf.constant(128, dtype=tf.int32).numpy()
    beta = 7.5
    dtype = tf.float32
    name = "window7"
    input_dict = {"window_length": window_length, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid
    window_length = tf.constant(2048, dtype=tf.int32).numpy()
    beta = 10.0
    dtype = tf.float64
    name = "window8"
    input_dict = {"window_length": window_length, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid
    window_length = tf.constant(7, dtype=tf.int32).numpy()
    beta = 3.0
    dtype = tf.float32
    name = "window9"
    input_dict = {"window_length": window_length, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid
    window_length = tf.constant(4096, dtype=tf.int32).numpy()
    beta = 1.0
    dtype = tf.float64
    name = "window10"
    input_dict = {"window_length": window_length, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.signal.kaiser_bessel_derived_window"] = tf_signal_kaiser_bessel_derived_window_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.kaiser_bessel_derived_window' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.kaiser_bessel_derived_window'.")

check_valid('tf.signal.kaiser_bessel_derived_window', generated_inputs['tf.signal.kaiser_bessel_derived_window'], lib="tf", suffix=0)
