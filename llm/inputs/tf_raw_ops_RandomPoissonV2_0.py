
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_RandomPoissonV2_inputs():
    list_of_inputs = []

    # Input 1
    shape = np.array([5], dtype=np.int32)
    rate = np.array([2.0], dtype=np.float32)
    seed = 10
    seed2 = 20
    dtype = tf.int64
    name = "poisson_sample_1"
    input_dict = {"shape": shape, "rate": rate, "seed": seed, "seed2": seed2, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = np.array([2, 3], dtype=np.int64)
    rate = np.array([[1.0, 5.0, 10.0], [2.0, 7.0, 12.0]], dtype=np.float64)
    seed = 1
    seed2 = 1
    dtype = tf.float32
    name = "poisson_sample_2"
    input_dict = {"shape": shape, "rate": rate, "seed": seed, "seed2": seed2, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = np.array([10], dtype=np.int32)
    rate = np.array([0.5] * 10, dtype=np.float32)
    seed = 42
    seed2 = 123
    dtype = tf.int32
    name = "poisson_sample_3"
    input_dict = {"shape": shape, "rate": rate, "seed": seed, "seed2": seed2, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = np.array([2, 2, 2], dtype=np.int64)
    rate = np.array([[[1.5, 2.5], [3.5, 4.5]], [[5.5, 6.5], [7.5, 8.5]]], dtype=np.float64)
    seed = 1
    seed2 = 2
    dtype = tf.int64
    name = "poisson_sample_4"
    input_dict = {"shape": shape, "rate": rate, "seed": seed, "seed2": seed2, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    shape = np.array([1], dtype=np.int32)
    rate = np.array([15.0], dtype=np.float32)
    seed = 100
    seed2 = 200
    dtype = tf.int32
    name = "poisson_sample_5"
    input_dict = {"shape": shape, "rate": rate, "seed": seed, "seed2": seed2, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = np.array([3, 4], dtype=np.int64)
    rate = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]], dtype=np.float64)
    seed = 0
    seed2 = 1
    dtype = tf.float64
    name = "poisson_sample_6"
    input_dict = {"shape": shape, "rate": rate, "seed": seed, "seed2": seed2, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shape = np.array([7], dtype=np.int32)
    rate = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], dtype=np.float32)
    seed = 5
    seed2 = 6
    dtype = tf.half
    name = "poisson_sample_7"
    input_dict = {"shape": shape, "rate": rate, "seed": seed, "seed2": seed2, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = np.array([1, 1, 1, 1], dtype=np.int64)
    rate = np.array([[[[20.0]]]], dtype=np.float64)
    seed = 7
    seed2 = 8
    dtype = tf.int32
    name = "poisson_sample_8"
    input_dict = {"shape": shape, "rate": rate, "seed": seed, "seed2": seed2, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape = np.array([4], dtype=np.int32)
    rate = np.array([0.01, 0.05, 0.1, 0.5], dtype=np.float32)
    seed = 9
    seed2 = 10
    dtype = tf.float32
    name = "poisson_sample_9"
    input_dict = {"shape": shape, "rate": rate, "seed": seed, "seed2": seed2, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = np.array([2], dtype=np.int64)
    rate = np.array([50.0, 100.0], dtype=np.float64)
    seed = 11
    seed2 = 12
    dtype = tf.int64
    name = "poisson_sample_10"
    input_dict = {"shape": shape, "rate": rate, "seed": seed, "seed2": seed2, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RandomPoissonV2"] = tf_raw_ops_RandomPoissonV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RandomPoissonV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RandomPoissonV2'.")

check_valid('tf.raw_ops.RandomPoissonV2', generated_inputs['tf.raw_ops.RandomPoissonV2'], lib="tf", suffix=0)
