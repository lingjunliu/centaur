
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_RandomUniform_inputs():
    list_of_inputs = []

    # Input 1
    shape = np.array([2, 3], dtype=np.int32)
    dtype = tf.float32
    seed = 0
    seed2 = 0
    name = "uniform_1"
    input_dict = {"shape": shape, "dtype": dtype, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = np.array([5, 5], dtype=np.int64)
    dtype = tf.float64
    seed = 123
    seed2 = 456
    name = "uniform_2"
    input_dict = {"shape": shape, "dtype": dtype, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = np.array([10], dtype=np.int32)
    dtype = tf.bfloat16
    seed = -1
    seed2 = -2
    name = "uniform_3"
    input_dict = {"shape": shape, "dtype": dtype, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = np.array([1, 2, 3, 4], dtype=np.int64)
    dtype = tf.half
    seed = 1000
    seed2 = 2000
    name = "uniform_4"
    input_dict = {"shape": shape, "dtype": dtype, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = np.array([7, 1], dtype=np.int32)
    dtype = tf.float32
    seed = 0
    seed2 = 1
    name = "uniform_5"
    input_dict = {"shape": shape, "dtype": dtype, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = np.array([2, 2, 2], dtype=np.int64)
    dtype = tf.float64
    seed = 10
    seed2 = 20
    name = "uniform_6"
    input_dict = {"shape": shape, "dtype": dtype, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shape = np.array([3], dtype=np.int32)
    dtype = tf.bfloat16
    seed = -10
    seed2 = -20
    name = "uniform_7"
    input_dict = {"shape": shape, "dtype": dtype, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    shape = np.array([1, 1, 1, 1, 1], dtype=np.int64)
    dtype = tf.half
    seed = 12345
    seed2 = 67890
    name = "uniform_8"
    input_dict = {"shape": shape, "dtype": dtype, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape = np.array([4, 4], dtype=np.int32)
    dtype = tf.float32
    seed = 5
    seed2 = 5
    name = "uniform_9"
    input_dict = {"shape": shape, "dtype": dtype, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = np.array([6, 2], dtype=np.int64)
    dtype = tf.float64
    seed = 999
    seed2 = 111
    name = "uniform_10"
    input_dict = {"shape": shape, "dtype": dtype, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RandomUniform"] = tf_raw_ops_RandomUniform_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RandomUniform' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RandomUniform'.")

check_valid('tf.raw_ops.RandomUniform', generated_inputs['tf.raw_ops.RandomUniform'], lib="tf", suffix=0)
