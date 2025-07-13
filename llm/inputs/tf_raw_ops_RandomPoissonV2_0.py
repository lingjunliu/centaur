
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
    seed = 0
    seed2 = 0
    dtype = tf.int64
    name = "poisson_1"
    input_dict = {"shape": shape, "rate": rate, "seed": seed, "seed2": seed2, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = np.array([2, 3], dtype=np.int64)
    rate = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    seed = 123
    seed2 = 456
    dtype = tf.float32
    name = "poisson_2"
    input_dict = {"shape": shape, "rate": rate, "seed": seed, "seed2": seed2, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = np.array([10], dtype=np.int32)
    rate = np.array([0.5] * 10, dtype=np.float32)
    seed = 789
    seed2 = 101
    dtype = tf.int32
    name = "poisson_3"
    input_dict = {"shape": shape, "rate": rate, "seed": seed, "seed2": seed2, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = np.array([2, 2, 2], dtype=np.int64)
    rate = np.array([[[1.5, 2.5], [3.5, 4.5]], [[5.5, 6.5], [7.5, 8.5]]], dtype=np.float64)
    seed = 222
    seed2 = 333
    dtype = tf.int64
    name = "poisson_4"
    input_dict = {"shape": shape, "rate": rate, "seed": seed, "seed2": seed2, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = np.array([1], dtype=np.int32)
    rate = np.array([15.0], dtype=np.float32)
    seed = 444
    seed2 = 555
    dtype = tf.float64
    name = "poisson_5"
    input_dict = {"shape": shape, "rate": rate, "seed": seed, "seed2": seed2, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = np.array([4, 4], dtype=np.int64)
    rate = np.array([[1.0, 5.0, 10.0, 15.0],[2.0, 6.0, 11.0, 16.0],[3.0, 7.0, 12.0, 17.0],[4.0, 8.0, 13.0, 18.0]], dtype=np.float32)
    seed = 666
    seed2 = 777
    dtype = tf.int32
    name = "poisson_6"
    input_dict = {"shape": shape, "rate": rate, "seed": seed, "seed2": seed2, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    shape = np.array([3], dtype=np.int32)
    rate = np.array([0.1, 0.5, 1.0], dtype=np.float64)
    seed = 888
    seed2 = 999
    dtype = tf.float32
    name = "poisson_7"
    input_dict = {"shape": shape, "rate": rate, "seed": seed, "seed2": seed2, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = np.array([2, 5], dtype=np.int64)
    rate = np.array([[0.2, 1.2, 2.2, 3.2, 4.2],[5.2, 6.2, 7.2, 8.2, 9.2]], dtype=np.float32)
    seed = 1010
    seed2 = 1111
    dtype = tf.int64
    name = "poisson_8"
    input_dict = {"shape": shape, "rate": rate, "seed": seed, "seed2": seed2, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape = np.array([7], dtype=np.int32)
    rate = np.array([1.0] * 7, dtype=np.float32)
    seed = 1212
    seed2 = 1313
    dtype = tf.int32
    name = "poisson_9"
    input_dict = {"shape": shape, "rate": rate, "seed": seed, "seed2": seed2, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = np.array([1, 1, 1], dtype=np.int64)
    rate = np.array([[[100.0]]], dtype=np.float64)
    seed = 1414
    seed2 = 1515
    dtype = tf.float64
    name = "poisson_10"
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
