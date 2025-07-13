
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_normal_inputs():
    list_of_inputs = []

    # Input 1
    shape = np.array([2, 3], dtype=np.int32)
    seed = np.array([1, 2], dtype=np.int32)
    mean = 0.0
    stddev = 1.0
    dtype = np.float32
    name = "normal_1"
    alg = 'auto_select'
    input_dict = {"shape": shape, "seed": seed, "mean": mean, "stddev": stddev, "dtype": dtype, "name": name, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = np.array([5, 5], dtype=np.int32)
    seed = np.array([3, 4], dtype=np.int32)
    mean = 1.0
    stddev = 2.0
    dtype = np.float64
    name = "normal_2"
    alg = 'philox'
    input_dict = {"shape": shape, "seed": seed, "mean": mean, "stddev": stddev, "dtype": dtype, "name": name, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = np.array([10], dtype=np.int32)
    seed = np.array([5, 6], dtype=np.int32)
    mean = -1.0
    stddev = 0.5
    dtype = np.float16
    name = "normal_3"
    alg = 'threefry'
    input_dict = {"shape": shape, "seed": seed, "mean": mean, "stddev": stddev, "dtype": dtype, "name": name, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = np.array([1, 2, 3], dtype=np.int32)
    seed = np.array([7, 8], dtype=np.int32)
    mean = 0.5
    stddev = 1.5
    dtype = np.float32
    name = "normal_4"
    alg = 'auto_select'
    input_dict = {"shape": shape, "seed": seed, "mean": mean, "stddev": stddev, "dtype": dtype, "name": name, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = np.array([4, 4, 4], dtype=np.int32)
    seed = np.array([9, 10], dtype=np.int32)
    mean = -0.5
    stddev = 0.25
    dtype = np.float64
    name = "normal_5"
    alg = 'philox'
    input_dict = {"shape": shape, "seed": seed, "mean": mean, "stddev": stddev, "dtype": dtype, "name": name, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = np.array([2, 2, 2, 2], dtype=np.int32)
    seed = np.array([11, 12], dtype=np.int32)
    mean = 2.0
    stddev = 0.75
    dtype = np.float16
    name = "normal_6"
    alg = 'threefry'
    input_dict = {"shape": shape, "seed": seed, "mean": mean, "stddev": stddev, "dtype": dtype, "name": name, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shape = np.array([3, 5], dtype=np.int32)
    seed = np.array([13, 14], dtype=np.int32)
    mean = -2.5
    stddev = 1.25
    dtype = np.float32
    name = "normal_7"
    alg = 'auto_select'
    input_dict = {"shape": shape, "seed": seed, "mean": mean, "stddev": stddev, "dtype": dtype, "name": name, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = np.array([7], dtype=np.int32)
    seed = np.array([15, 16], dtype=np.int32)
    mean = 0.25
    stddev = 0.1
    dtype = np.float64
    name = "normal_8"
    alg = 'philox'

    input_dict = {"shape": shape, "seed": seed, "mean": mean, "stddev": stddev, "dtype": dtype, "name": name, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.stateless_normal"] = tf_random_stateless_normal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random.stateless_normal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_normal'.")

check_valid('tf.random.stateless_normal', generated_inputs['tf.random.stateless_normal'], lib="tf", suffix=0)
