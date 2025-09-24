
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_truncated_normal_inputs():
    list_of_inputs = []

    # Input 1
    shape = tf.constant([2, 3], dtype=tf.int32).numpy()
    mean = 0.0
    stddev = 1.0
    dtype = tf.float32
    seed = 123
    name = "truncated_normal_1"
    input_dict = {"shape": shape, "mean": mean, "stddev": stddev, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = tf.constant([5], dtype=tf.int32).numpy()
    mean = 5.0
    stddev = 2.0
    dtype = tf.float64
    seed = 456
    name = "truncated_normal_2"
    input_dict = {"shape": shape, "mean": mean, "stddev": stddev, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = tf.constant([1, 4, 2], dtype=tf.int32).numpy()
    mean = -2.0
    stddev = 0.5
    dtype = tf.float16
    seed = 789
    name = "truncated_normal_3"
    input_dict = {"shape": shape, "mean": mean, "stddev": stddev, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = tf.constant([], dtype=tf.int32).numpy()
    mean = 1.5
    stddev = 0.75
    dtype = tf.float32
    seed = 101
    name = "truncated_normal_4"
    input_dict = {"shape": shape, "mean": mean, "stddev": stddev, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = tf.constant([3, 3], dtype=tf.int32).numpy()
    mean = 0.0
    stddev = 1.0
    dtype = tf.float32
    seed = 1234
    name = None
    input_dict = {"shape": shape, "mean": mean, "stddev": stddev, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    shape = tf.constant([4, 1], dtype=tf.int32).numpy()
    mean = -1.0
    stddev = 0.5
    dtype = tf.float32
    seed = 111
    name = "truncated_normal_6"
    input_dict = {"shape": shape, "mean": mean, "stddev": stddev, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shape = tf.constant([2, 2, 2, 2], dtype=tf.int32).numpy()
    mean = 2.0
    stddev = 1.5
    dtype = tf.float64
    seed = 222
    name = "truncated_normal_7"
    input_dict = {"shape": shape, "mean": mean, "stddev": stddev, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = tf.constant([7], dtype=tf.int32).numpy()
    mean = -3.0
    stddev = 0.25
    dtype = tf.float16
    seed = 333
    name = "truncated_normal_8"
    input_dict = {"shape": shape, "mean": mean, "stddev": stddev, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape = tf.constant([1, 1, 1], dtype=tf.int32).numpy()
    mean = 0.5
    stddev = 0.1
    dtype = tf.float32
    seed = 444
    name = "truncated_normal_9"
    input_dict = {"shape": shape, "mean": mean, "stddev": stddev, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = tf.constant([10, 5], dtype=tf.int32).numpy()
    mean = 10.0
    stddev = 3.0
    dtype = tf.float64
    seed = 555
    name = "truncated_normal_10"
    input_dict = {"shape": shape, "mean": mean, "stddev": stddev, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.truncated_normal"] = tf_random_truncated_normal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random.truncated_normal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.truncated_normal'.")

check_valid('tf.random.truncated_normal', generated_inputs['tf.random.truncated_normal'], lib="tf", suffix=0)
