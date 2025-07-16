
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.experimental.numpy.experimental_enable_numpy_behavior()

def tf_experimental_numpy_arange_inputs():
    list_of_inputs = []

    # Input 1: Basic integer range
    start = tf.constant(0, dtype=tf.int32)
    stop = tf.constant(10, dtype=tf.int32)
    step = tf.constant(1, dtype=tf.int32)
    dtype = tf.int32
    input_dict = {"start": start, "stop": stop, "step": step, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2:  Range with a start value
    start = tf.constant(5, dtype=tf.int32)
    stop = tf.constant(15, dtype=tf.int32)
    step = tf.constant(2, dtype=tf.int32)
    dtype = tf.int32
    input_dict = {"start": start, "stop": stop, "step": step, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative step
    start = tf.constant(10, dtype=tf.int32)
    stop = tf.constant(0, dtype=tf.int32)
    step = tf.constant(-1, dtype=tf.int32)
    dtype = tf.int32
    input_dict = {"start": start, "stop": stop, "step": step, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4:  Float range
    start = tf.constant(0.0, dtype=tf.float32)
    stop = tf.constant(5.0, dtype=tf.float32)
    step = tf.constant(0.5, dtype=tf.float32)
    dtype = tf.float32
    input_dict = {"start": start, "stop": stop, "step": step, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5:  Negative float range and step
    start = tf.constant(5.0, dtype=tf.float32)
    stop = tf.constant(0.0, dtype=tf.float32)
    step = tf.constant(-0.5, dtype=tf.float32)
    dtype = tf.float32
    input_dict = {"start": start, "stop": stop, "step": step, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6:  Different dtypes for start, stop and step which can be cast to a largest dtype
    start = tf.constant(0, dtype=tf.int16)
    stop = tf.constant(10, dtype=tf.int32)
    step = tf.constant(1, dtype=tf.int8)
    dtype = tf.int64
    input_dict = {"start": start, "stop": stop, "step": step, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Only start specified.
    start = tf.constant(10, dtype=tf.int32)
    stop = tf.constant(11, dtype=tf.int32)
    step = tf.constant(1, dtype=tf.int32)
    dtype = tf.int32
    input_dict = {"start": start, "stop": stop, "step": step, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large range, int64
    start = tf.constant(0, dtype=tf.int64)
    stop = tf.constant(10000000000, dtype=tf.int64)
    step = tf.constant(100000000, dtype=tf.int64)
    dtype = tf.int64
    input_dict = {"start": start, "stop": stop, "step": step, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Start and stop as float, but dtype is int. Results will be floored.
    start = tf.constant(0.5, dtype=tf.float32)
    stop = tf.constant(5.8, dtype=tf.float32)
    step = tf.constant(1.0, dtype=tf.float32)
    dtype = tf.float32 #dtype must be float as well
    input_dict = {"start": start, "stop": stop, "step": step, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: start and stop as negative float, but dtype is int. Results will be floored.
    start = tf.constant(-5.5, dtype=tf.float32)
    stop = tf.constant(-0.5, dtype=tf.float32)
    step = tf.constant(1.0, dtype=tf.float32)
    dtype = tf.float32 #dtype must be float as well
    input_dict = {"start": start, "stop": stop, "step": step, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: step as float, make start, stop and dtype float
    start = tf.constant(0, dtype=tf.float32)
    stop = tf.constant(10, dtype=tf.float32)
    step = tf.constant(2.5, dtype=tf.float32)
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
