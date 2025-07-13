
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_RandomUniformInt_inputs():
    list_of_inputs = []

    # Input 1
    shape = np.array([5], dtype=np.int32)
    minval = np.array(0, dtype=np.int32)
    maxval = np.array(10, dtype=np.int32)
    seed = 0
    seed2 = 0
    name = "random_uniform_int_1"
    input_dict = {"shape": tf.constant(shape), "minval": tf.constant(minval), "maxval": tf.constant(maxval, dtype=tf.int32), "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = np.array([2, 3], dtype=np.int32)
    minval = np.array(-5, dtype=np.int32)
    maxval = np.array(5, dtype=np.int32)
    seed = 1
    seed2 = 2
    name = "random_uniform_int_2"
    input_dict = {"shape": tf.constant(shape), "minval": tf.constant(minval), "maxval": tf.constant(maxval, dtype=tf.int32), "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = np.array([100], dtype=np.int64)
    minval = np.array(100, dtype=np.int64)
    maxval = np.array(200, dtype=np.int64)
    seed = 123
    seed2 = 456
    name = "random_uniform_int_3"
    input_dict = {"shape": tf.constant(shape), "minval": tf.constant(minval), "maxval": tf.constant(maxval, dtype=tf.int64), "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = np.array([2, 2, 2], dtype=np.int32)
    minval = np.array(-100, dtype=np.int32)
    maxval = np.array(0, dtype=np.int32)
    seed = 789
    seed2 = 101
    name = "random_uniform_int_4"
    input_dict = {"shape": tf.constant(shape), "minval": tf.constant(minval), "maxval": tf.constant(maxval, dtype=tf.int32), "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = np.array([5, 5], dtype=np.int64)
    minval = np.array(0, dtype=np.int64)
    maxval = np.array(1, dtype=np.int64)
    seed = 112
    seed2 = 131
    name = "random_uniform_int_5"
    input_dict = {"shape": tf.constant(shape), "minval": tf.constant(minval), "maxval": tf.constant(maxval, dtype=tf.int64), "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    shape = np.array([1], dtype=np.int32)
    minval = np.array(-2**7, dtype=np.int32)
    maxval = np.array(2**7-1, dtype=np.int32)
    seed = 0
    seed2 = 0
    name = "random_uniform_int_6"
    input_dict = {"shape": tf.constant(shape), "minval": tf.constant(minval), "maxval": tf.constant(maxval, dtype=tf.int32), "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shape = np.array([2,5,3], dtype=np.int32)
    minval = np.array(50, dtype=np.int32)
    maxval = np.array(51, dtype=np.int32)
    seed = 5
    seed2 = 6
    name = "random_uniform_int_7"
    input_dict = {"shape": tf.constant(shape), "minval": tf.constant(minval), "maxval": tf.constant(maxval, dtype=tf.int32), "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = np.array([1,1,1,1,1], dtype=np.int64)
    minval = np.array(1, dtype=np.int64)
    maxval = np.array(2, dtype=np.int64)
    seed = 7
    seed2 = 8
    name = "random_uniform_int_8"
    input_dict = {"shape": tf.constant(shape), "minval": tf.constant(minval), "maxval": tf.constant(maxval, dtype=tf.int64), "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape = np.array([0], dtype=np.int32)
    minval = np.array(0, dtype=np.int32)
    maxval = np.array(10, dtype=np.int32)
    seed = 9
    seed2 = 10
    name = "random_uniform_int_9"
    input_dict = {"shape": tf.constant(shape), "minval": tf.constant(minval), "maxval": tf.constant(maxval, dtype=tf.int32), "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = np.array([2, 3], dtype=np.int64)
    minval = np.array(-5, dtype=np.int64)
    maxval = np.array(5, dtype=np.int64)
    seed = 1
    seed2 = 2
    name = "random_uniform_int_10"
    input_dict = {"shape": tf.constant(shape), "minval": tf.constant(minval), "maxval": tf.constant(maxval, dtype=tf.int64), "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    shape = np.array([10], dtype=np.int64)
    minval = np.array(1, dtype=np.int64)
    maxval = np.array(5, dtype=np.int64)
    seed = 1
    seed2 = 2
    name = "random_uniform_int_11"
    input_dict = {"shape": tf.constant(shape), "minval": tf.constant(minval), "maxval": tf.constant(maxval, dtype=tf.int64), "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    shape = np.array([2,2], dtype=np.int32)
    minval = np.array(-10, dtype=np.int32)
    maxval = np.array(10, dtype=np.int32)
    seed = 13
    seed2 = 14
    name = "random_uniform_int_12"
    input_dict = {"shape": tf.constant(shape), "minval": tf.constant(minval), "maxval": tf.constant(maxval, dtype=tf.int32), "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RandomUniformInt"] = tf_raw_ops_RandomUniformInt_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RandomUniformInt' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RandomUniformInt'.")

check_valid('tf.raw_ops.RandomUniformInt', generated_inputs['tf.raw_ops.RandomUniformInt'], lib="tf", suffix=0)
