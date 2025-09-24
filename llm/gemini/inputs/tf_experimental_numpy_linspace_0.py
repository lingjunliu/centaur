
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_linspace_inputs():
    list_of_inputs = []

    # Input 1
    start = tf.constant(0.0, dtype=tf.float32)
    stop = tf.constant(1.0, dtype=tf.float32)
    num = 5
    endpoint = True
    retstep = False
    dtype = tf.float32
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "endpoint": endpoint, "retstep": retstep, "dtype": dtype, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    start = tf.constant(1.0, dtype=tf.float64)
    stop = tf.constant(10.0, dtype=tf.float64)
    num = 10
    endpoint = False
    retstep = True
    dtype = tf.float64
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "endpoint": endpoint, "retstep": retstep, "dtype": dtype, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    start = tf.constant(-5.0, dtype=tf.float32)
    stop = tf.constant(5.0, dtype=tf.float32)
    num = 20
    endpoint = True
    retstep = False
    dtype = tf.float32
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "endpoint": endpoint, "retstep": retstep, "dtype": dtype, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    start = tf.constant([1.0, 2.0], dtype=tf.float32)
    stop = tf.constant([3.0, 4.0], dtype=tf.float32)
    num = 5
    endpoint = True
    retstep = False
    dtype = tf.float32
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "endpoint": endpoint, "retstep": retstep, "dtype": dtype, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    start = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float64)
    stop = tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float64)
    num = 3
    endpoint = False
    retstep = True
    dtype = tf.float64
    axis = 1
    input_dict = {"start": start, "stop": stop, "num": num, "endpoint": endpoint, "retstep": retstep, "dtype": dtype, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    start = tf.constant(0, dtype=tf.int32)
    stop = tf.constant(10, dtype=tf.int32)
    num = 6
    endpoint = True
    retstep = False
    dtype = tf.int32
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "endpoint": endpoint, "retstep": retstep, "dtype": dtype, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    start = tf.constant([-10, -5], dtype=tf.int32)
    stop = tf.constant([0, 5], dtype=tf.int32)
    num = 4
    endpoint = False
    retstep = False
    dtype = tf.int32
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "endpoint": endpoint, "retstep": retstep, "dtype": dtype, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    start = tf.constant(1.0, dtype=tf.float32)
    stop = tf.constant(10.0, dtype=tf.float32)
    num = 1
    endpoint = True
    retstep = False
    dtype = tf.float32
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "endpoint": endpoint, "retstep": retstep, "dtype": dtype, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    start = tf.constant(0.0, dtype=tf.float32)
    stop = tf.constant(1.0, dtype=tf.float32)
    num = 5
    endpoint = False
    retstep = False
    dtype = tf.float32
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "endpoint": endpoint, "retstep": retstep, "dtype": dtype, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    start = tf.constant(5, dtype=tf.int32)
    stop = tf.constant(1, dtype=tf.int32)
    num = 5
    endpoint = True
    retstep = False
    dtype = tf.int32
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "endpoint": endpoint, "retstep": retstep, "dtype": dtype, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.linspace"] = tf_experimental_numpy_linspace_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.linspace' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.linspace'.")

check_valid('tf.experimental.numpy.linspace', generated_inputs['tf.experimental.numpy.linspace'], lib="tf", suffix=0)
