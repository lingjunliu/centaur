
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_uniform_inputs():
    list_of_inputs = []

    # Input 1
    shape = np.array([2, 3], dtype=np.int32)
    seed = np.array([1, 2], dtype=np.int32)
    minval = np.array(0.0, dtype=np.float32)
    maxval = np.array(1.0, dtype=np.float32)
    dtype = tf.float32
    name = "uniform1"
    alg = "auto_select"

    input_dict = {
        "shape": shape,
        "seed": seed,
        "minval": minval,
        "maxval": maxval,
        "dtype": dtype,
        "name": name,
        "alg": alg
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = np.array([5], dtype=np.int32)
    seed = np.array([3, 4], dtype=np.int32)
    minval = np.array(-1.0, dtype=np.float32)
    maxval = np.array(1.0, dtype=np.float32)
    dtype = tf.float32
    name = "uniform2"
    alg = "philox"

    input_dict = {
        "shape": shape,
        "seed": seed,
        "minval": minval,
        "maxval": maxval,
        "dtype": dtype,
        "name": name,
        "alg": alg
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = np.array([2, 2, 2], dtype=np.int32)
    seed = np.array([5, 6], dtype=np.int32)
    minval = np.array(2.0, dtype=np.float32)
    maxval = np.array(5.0, dtype=np.float32)
    dtype = tf.float32
    name = "uniform3"
    alg = "threefry"

    input_dict = {
        "shape": shape,
        "seed": seed,
        "minval": minval,
        "maxval": maxval,
        "dtype": dtype,
        "name": name,
        "alg": alg
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 - int32
    shape = np.array([4], dtype=np.int32)
    seed = np.array([7, 8], dtype=np.int32)
    minval = np.array(1, dtype=np.int32)
    maxval = np.array(10, dtype=np.int32)
    dtype = tf.int32
    name = "uniform4"
    alg = "auto_select"

    input_dict = {
        "shape": shape,
        "seed": seed,
        "minval": minval,
        "maxval": maxval,
        "dtype": dtype,
        "name": name,
        "alg": alg
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 - int64
    shape = np.array([3, 3], dtype=np.int32)
    seed = np.array([9, 10], dtype=np.int32)
    minval = np.array(-5, dtype=np.int64)
    maxval = np.array(5, dtype=np.int64)
    dtype = tf.int64
    name = "uniform5"
    alg = "philox"

    input_dict = {
        "shape": shape,
        "seed": seed,
        "minval": minval,
        "maxval": maxval,
        "dtype": dtype,
        "name": name,
        "alg": alg
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 - float64
    shape = np.array([1, 5], dtype=np.int32)
    seed = np.array([11, 12], dtype=np.int32)
    minval = np.array(0.0, dtype=np.float64)
    maxval = np.array(1.0, dtype=np.float64)
    dtype = tf.float64
    name = "uniform6"
    alg = "threefry"

    input_dict = {
        "shape": shape,
        "seed": seed,
        "minval": minval,
        "maxval": maxval,
        "dtype": dtype,
        "name": name,
        "alg": alg
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - full range int32
    shape = np.array([6], dtype=np.int32)
    seed = np.array([13, 14], dtype=np.int32)
    minval = None
    maxval = None
    dtype = tf.int32
    name = "uniform7"
    alg = "auto_select"

    input_dict = {
        "shape": shape,
        "seed": seed,
        "minval": minval,
        "maxval": maxval,
        "dtype": dtype,
        "name": name,
        "alg": alg
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    # Input 9 - negative minval, maxval
    shape = np.array([5], dtype=np.int32)
    seed = np.array([3, 4], dtype=np.int32)
    minval = np.array(-5.0, dtype=np.float32)
    maxval = np.array(-1.0, dtype=np.float32)
    dtype = tf.float32
    name = "uniform9"
    alg = "philox"

    input_dict = {
        "shape": shape,
        "seed": seed,
        "minval": minval,
        "maxval": maxval,
        "dtype": dtype,
        "name": name,
        "alg": alg
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - different min/max vals
    shape = np.array([2, 2], dtype=np.int32)
    seed = np.array([1, 2], dtype=np.int32)
    minval = np.array(5.0, dtype=np.float32)
    maxval = np.array(10.0, dtype=np.float32)
    dtype = tf.float32
    name = "uniform10"
    alg = "auto_select"

    input_dict = {
        "shape": shape,
        "seed": seed,
        "minval": minval,
        "maxval": maxval,
        "dtype": dtype,
        "name": name,
        "alg": alg
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 - Different seed values
    shape = np.array([2, 2], dtype=np.int32)
    seed = np.array([12345, 67890], dtype=np.int32)
    minval = np.array(0.0, dtype=np.float32)
    maxval = np.array(1.0, dtype=np.float32)
    dtype = tf.float32
    name = "uniform11"
    alg = "auto_select"

    input_dict = {
        "shape": shape,
        "seed": seed,
        "minval": minval,
        "maxval": maxval,
        "dtype": dtype,
        "name": name,
        "alg": alg
    }
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
