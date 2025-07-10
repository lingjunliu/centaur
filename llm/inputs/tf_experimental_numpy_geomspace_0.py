
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_geomspace_inputs():
    list_of_inputs = []

    # Input 1
    start = tf.constant(1.0)
    stop = tf.constant(1000.0)
    num = 5
    endpoint = True
    dtype = tf.float32
    axis = 0

    input_dict = {
        "start": start,
        "stop": stop,
        "num": num,
        "endpoint": endpoint,
        "dtype": dtype,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    start = tf.constant(1.0)
    stop = tf.constant(1000.0)
    num = 5
    endpoint = False
    dtype = tf.float64
    axis = 0

    input_dict = {
        "start": start,
        "stop": stop,
        "num": num,
        "endpoint": endpoint,
        "dtype": dtype,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    start = tf.constant(1.0)
    stop = tf.constant(1000.0)
    num = 10
    endpoint = True
    dtype = tf.float32
    axis = 0

    input_dict = {
        "start": start,
        "stop": stop,
        "num": num,
        "endpoint": endpoint,
        "dtype": dtype,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    start = tf.constant(1.0)
    stop = tf.constant(1000.0)
    num = 10
    endpoint = False
    dtype = tf.float64
    axis = 0

    input_dict = {
        "start": start,
        "stop": stop,
        "num": num,
        "endpoint": endpoint,
        "dtype": dtype,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    start = tf.constant(0.1)
    stop = tf.constant(100.0)
    num = 7
    endpoint = True
    dtype = tf.float32
    axis = 0

    input_dict = {
        "start": start,
        "stop": stop,
        "num": num,
        "endpoint": endpoint,
        "dtype": dtype,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    start = tf.constant(0.1)
    stop = tf.constant(100.0)
    num = 7
    endpoint = False
    dtype = tf.float64
    axis = 0

    input_dict = {
        "start": start,
        "stop": stop,
        "num": num,
        "endpoint": endpoint,
        "dtype": dtype,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    start = tf.constant(-1.0)
    stop = tf.constant(-1000.0)
    num = 5
    endpoint = True
    dtype = tf.float32
    axis = 0

    input_dict = {
        "start": start,
        "stop": stop,
        "num": num,
        "endpoint": endpoint,
        "dtype": dtype,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    start = tf.constant(-1.0)
    stop = tf.constant(-1000.0)
    num = 5
    endpoint = False
    dtype = tf.float64
    axis = 0

    input_dict = {
        "start": start,
        "stop": stop,
        "num": num,
        "endpoint": endpoint,
        "dtype": dtype,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    start = tf.constant(2.0)
    stop = tf.constant(32.0)
    num = 6
    endpoint = True
    dtype = tf.float32
    axis = 0

    input_dict = {
        "start": start,
        "stop": stop,
        "num": num,
        "endpoint": endpoint,
        "dtype": dtype,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    start = tf.constant(2.0)
    stop = tf.constant(32.0)
    num = 6
    endpoint = False
    dtype = tf.float64
    axis = 0

    input_dict = {
        "start": start,
        "stop": stop,
        "num": num,
        "endpoint": endpoint,
        "dtype": dtype,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.geomspace"] = tf_experimental_numpy_geomspace_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.geomspace' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.geomspace'.")

check_valid('tf.experimental.numpy.geomspace', generated_inputs['tf.experimental.numpy.geomspace'], lib="tf", suffix=0)
