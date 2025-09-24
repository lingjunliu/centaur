
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_normal_inputs():
    list_of_inputs = []

    # Input 1
    shape = np.array([2, 3], dtype=np.int32)
    mean = np.array(0.0, dtype=np.float32)
    stddev = np.array(1.0, dtype=np.float32)
    dtype = tf.float32
    seed = 1
    name = "normal_1"
    input_dict = {"shape": shape, "mean": mean, "stddev": stddev, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = np.array([5], dtype=np.int32)
    mean = np.array(-1.0, dtype=np.float32)
    stddev = np.array(0.5, dtype=np.float32)
    dtype = tf.float32
    seed = 2
    name = "normal_2"
    input_dict = {"shape": shape, "mean": mean, "stddev": stddev, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = np.array([1, 4], dtype=np.int32)
    mean = np.array(2.0, dtype=np.float32)
    stddev = np.array(2.0, dtype=np.float32)
    dtype = tf.float32
    seed = 3
    name = "normal_3"
    input_dict = {"shape": shape, "mean": mean, "stddev": stddev, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = np.array([2, 2, 2], dtype=np.int32)
    mean = np.array(0.0, dtype=np.float64)
    stddev = np.array(1.0, dtype=np.float64)
    dtype = tf.float64
    seed = 4
    name = "normal_4"
    input_dict = {"shape": shape, "mean": mean, "stddev": stddev, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = np.array([3, 1], dtype=np.int32)
    mean = np.array(-2.5, dtype=np.float32)
    stddev = np.array(0.75, dtype=np.float32)
    dtype = tf.float32
    seed = 5
    name = "normal_5"
    input_dict = {"shape": shape, "mean": mean, "stddev": stddev, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = np.array([4, 4], dtype=np.int32)
    mean = np.array(0.0, dtype=np.float16)
    stddev = np.array(1.0, dtype=np.float16)
    dtype = tf.float16
    seed = 6
    name = "normal_6"
    input_dict = {"shape": shape, "mean": mean, "stddev": stddev, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shape = np.array([6], dtype=np.int32)
    mean = np.array(1.5, dtype=np.float32)
    stddev = np.array(0.25, dtype=np.float32)
    dtype = tf.float32
    seed = 7
    name = "normal_7"
    input_dict = {"shape": shape, "mean": mean, "stddev": stddev, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = np.array([1, 1, 1, 1], dtype=np.int32)
    mean = np.array(0.0, dtype=np.float64)
    stddev = np.array(1.0, dtype=np.float64)
    dtype = tf.float64
    seed = 8
    name = "normal_8"
    input_dict = {"shape": shape, "mean": mean, "stddev": stddev, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape = np.array([7, 2], dtype=np.int32)
    mean = np.array(-0.5, dtype=np.float32)
    stddev = np.array(1.5, dtype=np.float32)
    dtype = tf.float32
    seed = 9
    name = "normal_9"
    input_dict = {"shape": shape, "mean": mean, "stddev": stddev, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = np.array([8, 3, 2], dtype=np.int32)
    mean = np.array(1.0, dtype=np.float16)
    stddev = np.array(0.5, dtype=np.float16)
    dtype = tf.float16
    seed = 10
    name = "normal_10"
    input_dict = {"shape": shape, "mean": mean, "stddev": stddev, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    shape = np.array([10], dtype=np.int32)
    mean = np.array(-2.0, dtype=np.float64)
    stddev = np.array(2.5, dtype=np.float64)
    dtype = tf.float64
    seed = 11
    name = "normal_11"
    input_dict = {"shape": shape, "mean": mean, "stddev": stddev, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.normal"] = tf_random_normal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random.normal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.normal'.")

check_valid('tf.random.normal', generated_inputs['tf.random.normal'], lib="tf", suffix=0)
