
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_normal_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    shape = np.array([2, 3], dtype=np.int32)
    seed = np.array([1, 2], dtype=np.int32)
    mean = 0.0
    stddev = 1.0
    dtype = tf.float32
    name = "normal_1"
    alg = "auto_select"
    input_dict = {"shape": shape, "seed": seed, "mean": mean, "stddev": stddev, "dtype": dtype, "name": name, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shape, seed, and name
    shape = np.array([5, 4, 2], dtype=np.int32)
    seed = np.array([10, 20], dtype=np.int32)
    mean = 2.0
    stddev = 0.5
    dtype = tf.float32
    name = "normal_2"
    alg = "auto_select"
    input_dict = {"shape": shape, "seed": seed, "mean": mean, "stddev": stddev, "dtype": dtype, "name": name, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative mean and stddev
    shape = np.array([1, 1], dtype=np.int32)
    seed = np.array([-1, -2], dtype=np.int32)
    mean = -1.0
    stddev = 0.25
    dtype = tf.float32
    name = "normal_3"
    alg = "auto_select"
    input_dict = {"shape": shape, "seed": seed, "mean": mean, "stddev": stddev, "dtype": dtype, "name": name, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float64 dtype
    shape = np.array([4], dtype=np.int32)
    seed = np.array([5, 6], dtype=np.int32)
    mean = 1.5
    stddev = 0.75
    dtype = tf.float64
    name = "normal_4"
    alg = "auto_select"
    input_dict = {"shape": shape, "seed": seed, "mean": mean, "stddev": stddev, "dtype": dtype, "name": name, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Zero mean and small stddev
    shape = np.array([2, 2, 2, 2], dtype=np.int32)
    seed = np.array([7, 8], dtype=np.int32)
    mean = 0.0
    stddev = 0.001
    dtype = tf.float32
    name = "normal_5"
    alg = "auto_select"
    input_dict = {"shape": shape, "seed": seed, "mean": mean, "stddev": stddev, "dtype": dtype, "name": name, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger shape
    shape = np.array([100], dtype=np.int32)
    seed = np.array([9, 10], dtype=np.int32)
    mean = 5.0
    stddev = 2.0
    dtype = tf.float32
    name = "normal_6"
    alg = "auto_select"
    input_dict = {"shape": shape, "seed": seed, "mean": mean, "stddev": stddev, "dtype": dtype, "name": name, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: Float16 dtype
    shape = np.array([3, 2], dtype=np.int32)
    seed = np.array([13, 14], dtype=np.int32)
    mean = 0.0
    stddev = 1.0
    dtype = tf.float16
    name = "normal_8"
    alg = "auto_select"
    input_dict = {"shape": shape, "seed": seed, "mean": mean, "stddev": stddev, "dtype": dtype, "name": name, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: alg = "philox"
    shape = np.array([2, 3], dtype=np.int32)
    seed = np.array([1, 2], dtype=np.int32)
    mean = 0.0
    stddev = 1.0
    dtype = tf.float32
    name = "normal_9"
    alg = "philox"
    input_dict = {"shape": shape, "seed": seed, "mean": mean, "stddev": stddev, "dtype": dtype, "name": name, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: alg = "threefry"
    shape = np.array([2, 3], dtype=np.int32)
    seed = np.array([1, 2], dtype=np.int32)
    mean = 0.0
    stddev = 1.0
    dtype = tf.float32
    name = "normal_10"
    alg = "threefry"
    input_dict = {"shape": shape, "seed": seed, "mean": mean, "stddev": stddev, "dtype": dtype, "name": name, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different seed values
    shape = np.array([2, 3], dtype=np.int32)
    seed = np.array([12345, 67890], dtype=np.int32)
    mean = 0.0
    stddev = 1.0
    dtype = tf.float32
    name = "normal_11"
    alg = "auto_select"
    input_dict = {"shape": shape, "seed": seed, "mean": mean, "stddev": stddev, "dtype": dtype, "name": name, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Remove shape [1, 1]
    # shape = np.array([1, 1], dtype=np.int32)
    # seed = np.array([1, 2], dtype=np.int32)
    # mean = 0.0
    # stddev = 1.0
    # dtype = tf.float32
    # name = "normal_12"
    # alg = "auto_select"
    # input_dict = {"shape": shape, "seed": seed, "mean": mean, "stddev": stddev, "dtype": dtype, "name": name, "alg": alg}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: shape [2, 2], seed [1,2]
    shape = np.array([2, 2], dtype=np.int32)
    seed = np.array([1, 2], dtype=np.int32)
    mean = 0.0
    stddev = 1.0
    dtype = tf.float32
    name = "normal_13"
    alg = "auto_select"
    input_dict = {"shape": shape, "seed": seed, "mean": mean, "stddev": stddev, "dtype": dtype, "name": name, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13: Seed as numpy int32 array of shape (2,)
    shape = np.array([2, 3], dtype=np.int32)
    seed = np.array([1, 2], dtype=np.int32)
    mean = 0.0
    stddev = 1.0
    dtype = tf.float32
    name = "normal_14"
    alg = "auto_select"
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
