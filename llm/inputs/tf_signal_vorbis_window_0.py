
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_vorbis_window_inputs():
    list_of_inputs = []

    # Input 1: Basic case with int32
    window_length = tf.constant(10, dtype=tf.int32).numpy()
    dtype = tf.float32
    name = "vorbis_window_1"
    input_dict = {"window_length": window_length, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different window length with int64
    window_length = tf.constant(256, dtype=tf.int64).numpy()
    dtype = tf.float64
    name = "vorbis_window_2"
    input_dict = {"window_length": window_length, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3:  dtype float16
    window_length = tf.constant(64, dtype=tf.int32).numpy()
    dtype = tf.float16
    name = "vorbis_window_3"
    input_dict = {"window_length": window_length, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4:  dtype float32, different name
    window_length = tf.constant(128, dtype=tf.int32).numpy()
    dtype = tf.float32
    name = "my_vorbis_window"
    input_dict = {"window_length": window_length, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small window length
    window_length = tf.constant(2, dtype=tf.int32).numpy()
    dtype = tf.float32
    name = "vorbis_window_5"
    input_dict = {"window_length": window_length, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large window length
    window_length = tf.constant(2048, dtype=tf.int32).numpy()
    dtype = tf.float32
    name = "vorbis_window_6"
    input_dict = {"window_length": window_length, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float64 and a name with numbers
    window_length = tf.constant(512, dtype=tf.int32).numpy()
    dtype = tf.float64
    name = "vorbis_window_7_with_123"
    input_dict = {"window_length": window_length, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32 and a longer name
    window_length = tf.constant(32, dtype=tf.int32).numpy()
    dtype = tf.float32
    name = "a_very_long_vorbis_window_name"
    input_dict = {"window_length": window_length, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32 no name
    window_length = tf.constant(16, dtype=tf.int32).numpy()
    dtype = tf.float32
    name = None
    input_dict = {"window_length": window_length, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Window length is 1
    window_length = tf.constant(1, dtype=tf.int32).numpy()
    dtype = tf.float32
    name = "vorbis_window_10"
    input_dict = {"window_length": window_length, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.signal.vorbis_window"] = tf_signal_vorbis_window_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.vorbis_window' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.vorbis_window'.")

check_valid('tf.signal.vorbis_window', generated_inputs['tf.signal.vorbis_window'], lib="tf", suffix=0)
