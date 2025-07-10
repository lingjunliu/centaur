
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_sobol_sample_inputs():
    list_of_inputs = []

    # Input 1
    dim = tf.constant(2).numpy()
    num_results = 5
    skip = 0
    dtype = tf.float32
    name = "sobol_sample_1"
    input_dict = {"dim": dim, "num_results": num_results, "skip": skip, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    dim = tf.constant(5).numpy()
    num_results = 10
    skip = 10
    dtype = tf.float32
    name = "sobol_sample_2"
    input_dict = {"dim": dim, "num_results": num_results, "skip": skip, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    dim = tf.constant(1).numpy()
    num_results = 1
    skip = 0
    dtype = tf.float64
    name = "sobol_sample_3"
    input_dict = {"dim": dim, "num_results": num_results, "skip": skip, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    dim = tf.constant(10).numpy()
    num_results = 20
    skip = 5
    dtype = tf.float64
    name = "sobol_sample_4"
    input_dict = {"dim": dim, "num_results": num_results, "skip": skip, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    dim = tf.constant(3).numpy()
    num_results = 7
    skip = 2
    dtype = tf.float32
    name = "sobol_sample_5"
    input_dict = {"dim": dim, "num_results": num_results, "skip": skip, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    dim = tf.constant(15).numpy()
    num_results = 30
    skip = 15
    dtype = tf.float64
    name = "sobol_sample_6"
    input_dict = {"dim": dim, "num_results": num_results, "skip": skip, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    dim = tf.constant(4).numpy()
    num_results = 9
    skip = 1
    dtype = tf.float32
    name = "sobol_sample_7"
    input_dict = {"dim": dim, "num_results": num_results, "skip": skip, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    dim = tf.constant(7).numpy()
    num_results = 12
    skip = 3
    dtype = tf.float64
    name = "sobol_sample_8"
    input_dict = {"dim": dim, "num_results": num_results, "skip": skip, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    dim = tf.constant(6).numpy()
    num_results = 8
    skip = 4
    dtype = tf.float32
    name = "sobol_sample_9"
    input_dict = {"dim": dim, "num_results": num_results, "skip": skip, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    dim = tf.constant(8).numpy()
    num_results = 16
    skip = 8
    dtype = tf.float64
    name = "sobol_sample_10"
    input_dict = {"dim": dim, "num_results": num_results, "skip": skip, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.sobol_sample"] = tf_math_sobol_sample_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.sobol_sample' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.sobol_sample'.")

check_valid('tf.math.sobol_sample', generated_inputs['tf.math.sobol_sample'], lib="tf", suffix=0)
