
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_arange_inputs():
    list_of_inputs = []

    # Input 1: Basic integer range
    start = tf.constant(0, dtype=tf.int32)
    stop = tf.constant(10, dtype=tf.int32)
    step = tf.constant(1, dtype=tf.int32)
    dtype = tf.int32
    input_dict = {"start": start, "stop": stop, "step": step, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative start and stop
    start = tf.constant(-5, dtype=tf.int32)
    stop = tf.constant(5, dtype=tf.int32)
    step = tf.constant(2, dtype=tf.int32)
    dtype = tf.int32
    input_dict = {"start": start, "stop": stop, "step": step, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float range
    start = tf.constant(0.0, dtype=tf.float32)
    stop = tf.constant(1.0, dtype=tf.float32)
    step = tf.constant(0.1, dtype=tf.float32)
    dtype = tf.float32
    input_dict = {"start": start, "stop": stop, "step": step, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative float range
    start = tf.constant(-1.0, dtype=tf.float32)
    stop = tf.constant(1.0, dtype=tf.float32)
    step = tf.constant(0.2, dtype=tf.float32)
    dtype = tf.float32
    input_dict = {"start": start, "stop": stop, "step": step, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.arange"] = tf_experimental_numpy_arange_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.arange' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.arange'.")

check_valid('tf.experimental.numpy.arange', generated_inputs['tf.experimental.numpy.arange'], lib="tf", suffix=0)
