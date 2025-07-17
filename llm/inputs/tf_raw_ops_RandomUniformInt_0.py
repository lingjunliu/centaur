
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_random_uniform_int_inputs():
    list_of_inputs = []

    # Input 1
    shape = np.array([5], dtype=np.int32)
    minval = np.array(0, dtype=np.int32)
    maxval = np.array(10, dtype=np.int32)
    seed = 1
    seed2 = 0
    name = None

    input_dict = {
        "shape": shape,
        "minval": minval,
        "maxval": maxval,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = np.array([2, 3], dtype=np.int32)
    minval = np.array(-5, dtype=np.int32)
    maxval = np.array(5, dtype=np.int32)
    seed = 1
    seed2 = 2
    name = "random_uniform_int_1"

    input_dict = {
        "shape": shape,
        "minval": minval,
        "maxval": maxval,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = np.array([100], dtype=np.int32)
    minval = np.array(1000, dtype=np.int32)
    maxval = np.array(1001, dtype=np.int32)
    seed = 123
    seed2 = 456
    name = "random_uniform_int_2"

    input_dict = {
        "shape": shape,
        "minval": minval,
        "maxval": maxval,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    shape = np.array([2, 2, 2], dtype=np.int32)
    minval = np.array(0, dtype=np.int32)
    maxval = np.array(2, dtype=np.int32)
    seed = 789
    seed2 = 1011
    name = "random_uniform_int_3"

    input_dict = {
        "shape": shape,
        "minval": minval,
        "maxval": maxval,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    shape = np.array([5], dtype=np.int64)
    minval = np.array(0, dtype=np.int64)
    maxval = np.array(10, dtype=np.int64)
    seed = 1
    seed2 = 0
    name = None

    input_dict = {
        "shape": shape,
        "minval": minval,
        "maxval": maxval,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = np.array([2, 3], dtype=np.int64)
    minval = np.array(-5, dtype=np.int64)
    maxval = np.array(5, dtype=np.int64)
    seed = 1
    seed2 = 2
    name = "random_uniform_int_1_int64"

    input_dict = {
        "shape": shape,
        "minval": minval,
        "maxval": maxval,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shape = np.array([100], dtype=np.int64)
    minval = np.array(1000, dtype=np.int64)
    maxval = np.array(1001, dtype=np.int64)
    seed = 123
    seed2 = 456
    name = "random_uniform_int_2_int64"

    input_dict = {
        "shape": shape,
        "minval": minval,
        "maxval": maxval,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    shape = np.array([2, 2, 2], dtype=np.int64)
    minval = np.array(0, dtype=np.int64)
    maxval = np.array(2, dtype=np.int64)
    seed = 789
    seed2 = 1011
    name = "random_uniform_int_3_int64"

    input_dict = {
        "shape": shape,
        "minval": minval,
        "maxval": maxval,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape = np.array([1], dtype=np.int32)
    minval = np.array(-100, dtype=np.int32)
    maxval = np.array(100, dtype=np.int32)
    seed = 1
    seed2 = 2
    name = "random_uniform_int_4"

    input_dict = {
        "shape": shape,
        "minval": minval,
        "maxval": maxval,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = np.array([1], dtype=np.int64)
    minval = np.array(-100, dtype=np.int64)
    maxval = np.array(100, dtype=np.int64)
    seed = 1
    seed2 = 2
    name = "random_uniform_int_5"

    input_dict = {
        "shape": shape,
        "minval": minval,
        "maxval": maxval,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RandomUniformInt"] = tf_raw_ops_random_uniform_int_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RandomUniformInt' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RandomUniformInt'.")

check_valid('tf.raw_ops.RandomUniformInt', generated_inputs['tf.raw_ops.RandomUniformInt'], lib="tf", suffix=0)
