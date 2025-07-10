
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_logspace_inputs():
    list_of_inputs = []

    # Input 1
    start = tf.constant(0.0).numpy()
    stop = tf.constant(2.0).numpy()
    num = 5
    endpoint = True
    base = 10.0
    dtype = np.float32
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "endpoint": endpoint, "base": base, "dtype": dtype, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    start = tf.constant(-1.0).numpy()
    stop = tf.constant(1.0).numpy()
    num = 10
    endpoint = False
    base = 2.0
    dtype = np.float64
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "endpoint": endpoint, "base": base, "dtype": dtype, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    start = tf.constant(1.0).numpy()
    stop = tf.constant(5.0).numpy()
    num = 20
    endpoint = True
    base = np.e
    dtype = np.float32
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "endpoint": endpoint, "base": base, "dtype": dtype, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    start = tf.constant(-2.0).numpy()
    stop = tf.constant(0.0).numpy()
    num = 3
    endpoint = False
    base = 10.0
    dtype = np.float64
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "endpoint": endpoint, "base": base, "dtype": dtype, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    start = tf.constant(0.0, dtype=tf.float32).numpy()
    stop = tf.constant(3.0, dtype=tf.float32).numpy()
    num = 7
    endpoint = True
    base = 2.0
    dtype = np.float32
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "endpoint": endpoint, "base": base, "dtype": dtype, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    start = tf.constant(-3.0).numpy()
    stop = tf.constant(3.0).numpy()
    num = 12
    endpoint = False
    base = 10.0
    dtype = np.float64
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "endpoint": endpoint, "base": base, "dtype": dtype, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    start = tf.constant(2.0).numpy()
    stop = tf.constant(8.0).numpy()
    num = 18
    endpoint = True
    base = 2.0
    dtype = np.float32
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "endpoint": endpoint, "base": base, "dtype": dtype, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    start = tf.constant(-5.0).numpy()
    stop = tf.constant(5.0).numpy()
    num = 8
    endpoint = False
    base = np.e
    dtype = np.float64
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "endpoint": endpoint, "base": base, "dtype": dtype, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    start = tf.constant(1.5).numpy()
    stop = tf.constant(4.5).numpy()
    num = 6
    endpoint = True
    base = 10.0
    dtype = np.float32
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "endpoint": endpoint, "base": base, "dtype": dtype, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    start = tf.constant(-0.5).numpy()
    stop = tf.constant(0.5).numpy()
    num = 15
    endpoint = False
    base = 2.0
    dtype = np.float64
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "endpoint": endpoint, "base": base, "dtype": dtype, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.logspace"] = tf_experimental_numpy_logspace_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.logspace' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.logspace'.")

check_valid('tf.experimental.numpy.logspace', generated_inputs['tf.experimental.numpy.logspace'], lib="tf", suffix=0)
