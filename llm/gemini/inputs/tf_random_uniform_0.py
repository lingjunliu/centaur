
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_uniform_inputs():
    list_of_inputs = []

    # Input 1
    shape = tf.constant([2, 3], dtype=tf.int32)
    minval = tf.constant(0.0, dtype=tf.float32)
    maxval = tf.constant(1.0, dtype=tf.float32)
    dtype = tf.float32
    seed = 123
    name = "uniform_1"
    input_dict = {"shape": shape, "minval": minval, "maxval": maxval, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = tf.constant([5], dtype=tf.int32)
    minval = tf.constant(-1.0, dtype=tf.float32)
    maxval = tf.constant(1.0, dtype=tf.float32)
    dtype = tf.float32
    seed = 456
    name = "uniform_2"
    input_dict = {"shape": shape, "minval": minval, "maxval": maxval, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = tf.constant([], dtype=tf.int32)
    minval = tf.constant(5, dtype=tf.int32)
    maxval = tf.constant(10, dtype=tf.int32)
    dtype = tf.int32
    seed = 789
    name = "uniform_3"
    input_dict = {"shape": shape, "minval": minval, "maxval": maxval, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = tf.constant([4, 4], dtype=tf.int32)
    minval = tf.constant(-5.0, dtype=tf.float64)
    maxval = tf.constant(5.0, dtype=tf.float64)
    dtype = tf.float64
    seed = 101
    name = "uniform_4"
    input_dict = {"shape": shape, "minval": minval, "maxval": maxval, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = tf.constant([1, 2, 3], dtype=tf.int32)
    minval = tf.constant(0, dtype=tf.int64)
    maxval = tf.constant(100, dtype=tf.int64)
    dtype = tf.int64
    seed = 202
    name = "uniform_5"
    input_dict = {"shape": shape, "minval": minval, "maxval": maxval, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = tf.constant([2, 2, 2, 2], dtype=tf.int32)
    minval = tf.constant(-10.0, dtype=tf.float32)
    maxval = tf.constant(0.0, dtype=tf.float32)
    dtype = tf.float32
    seed = 303
    name = "uniform_6"
    input_dict = {"shape": shape, "minval": minval, "maxval": maxval, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    shape = tf.constant([7], dtype=tf.int32)
    minval = tf.constant(0, dtype=tf.int32)
    maxval = tf.constant(1, dtype=tf.int32)
    dtype = tf.int32
    seed = 404
    name = "uniform_7"
    input_dict = {"shape": shape, "minval": minval, "maxval": maxval, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = tf.constant([10, 1], dtype=tf.int32)
    minval = tf.constant(-100, dtype=tf.int64)
    maxval = tf.constant(-50, dtype=tf.int64)
    dtype = tf.int64
    seed = 505
    name = "uniform_8"
    input_dict = {"shape": shape, "minval": minval, "maxval": maxval, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape = tf.constant([3, 1, 5], dtype=tf.int32)
    minval = tf.constant(0.0, dtype=tf.float16)
    maxval = tf.constant(10.0, dtype=tf.float16)
    dtype = tf.float16
    seed = 606
    name = "uniform_9"
    input_dict = {"shape": shape, "minval": minval, "maxval": maxval, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = tf.constant([2], dtype=tf.int32)
    minval = tf.constant(-1.0, dtype=tf.float32)
    maxval = tf.constant(1.0, dtype=tf.float32)
    dtype = tf.float32
    seed = 707
    name = "uniform_10"
    input_dict = {"shape": shape, "minval": minval, "maxval": maxval, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    for i in range(len(list_of_inputs)):
        for key in list_of_inputs[i].keys():
            if isinstance(list_of_inputs[i][key], np.ndarray):
                list_of_inputs[i][key] = list_of_inputs[i][key]
            elif isinstance(list_of_inputs[i][key], tf.DType):
                list_of_inputs[i][key] = list_of_inputs[i][key].as_numpy_dtype
            elif isinstance(list_of_inputs[i][key], tf.Tensor):
                 list_of_inputs[i][key] = list_of_inputs[i][key].numpy()

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.uniform"] = tf_random_uniform_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random.uniform' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.uniform'.")

check_valid('tf.random.uniform', generated_inputs['tf.random.uniform'], lib="tf", suffix=0)
