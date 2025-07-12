
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_uniform_inputs():
    list_of_inputs = []

    # Input 1: Basic float32
    shape = np.array([2, 3], dtype=np.int32)
    seed = np.array([1, 2], dtype=np.int32)
    minval = np.array(0.0, dtype=np.float32)
    maxval = np.array(1.0, dtype=np.float32)
    dtype = tf.float32
    name = "uniform_float32"
    alg = "auto_select"
    input_dict = {"shape": shape, "seed": seed, "minval": minval, "maxval": maxval, "dtype": dtype, "name": name, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32 with explicit minval and maxval
    shape = np.array([5], dtype=np.int32)
    seed = np.array([3, 4], dtype=np.int32)
    minval = np.array(-10, dtype=np.int32)
    maxval = np.array(10, dtype=np.int32)
    dtype = tf.int32
    name = "uniform_int32"
    alg = "philox"
    input_dict = {"shape": shape, "seed": seed, "minval": minval, "maxval": maxval, "dtype": dtype, "name": name, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64 with a different shape and algorithm
    shape = np.array([1, 4, 2], dtype=np.int32)
    seed = np.array([5, 6], dtype=np.int32)
    minval = np.array(-2.5, dtype=np.float64)
    maxval = np.array(7.5, dtype=np.float64)
    dtype = tf.float64
    name = "uniform_float64"
    alg = "threefry"
    input_dict = {"shape": shape, "seed": seed, "minval": minval, "maxval": maxval, "dtype": dtype, "name": name, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64 with full range
    shape = np.array([3, 3], dtype=np.int32)
    seed = np.array([7, 8], dtype=np.int32)
    minval = None
    maxval = None
    dtype = tf.int64
    name = "uniform_int64_full_range"
    alg = "auto_select"
    input_dict = {"shape": shape, "seed": seed, "minval": minval, "maxval": maxval, "dtype": dtype, "name": name, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float16
    shape = np.array([4], dtype=np.int32)
    seed = np.array([9, 10], dtype=np.int32)
    minval = np.array(-1.0, dtype=np.float16)
    maxval = np.array(1.0, dtype=np.float16)
    dtype = tf.float16
    name = "uniform_float16"
    alg = "philox"
    input_dict = {"shape": shape, "seed": seed, "minval": minval, "maxval": maxval, "dtype": dtype, "name": name, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.stateless_uniform"] = tf_random_stateless_uniform_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random.stateless_uniform' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_uniform'.")

check_valid('tf.random.stateless_uniform', generated_inputs['tf.random.stateless_uniform'], lib="tf", suffix=0)
