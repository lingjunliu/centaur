
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_RandomGamma_inputs():
    list_of_inputs = []

    # Input 1
    shape = np.array([2, 3], dtype=np.int32)
    alpha = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32).reshape((2,3))
    seed = 1
    seed2 = 1
    name = "random_gamma_1"
    input_dict = {"shape": shape, "alpha": alpha, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = np.array([5], dtype=np.int64)
    alpha = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    seed = 123
    seed2 = 456
    name = "random_gamma_2"
    input_dict = {"shape": shape, "alpha": alpha, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = np.array([1, 4], dtype=np.int32)
    alpha = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).reshape((1,4))
    seed = 789
    seed2 = 101
    name = "random_gamma_3"
    input_dict = {"shape": shape, "alpha": alpha, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = np.array([3, 2, 2], dtype=np.int64)
    alpha = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1], dtype=np.float64).reshape((3,2,2))
    seed = 2
    seed2 = 3
    name = "random_gamma_4"
    input_dict = {"shape": shape, "alpha": alpha, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = np.array([4], dtype=np.int32)
    alpha = np.array([2.0, 2.1, 2.2, 2.3], dtype=np.float32)
    seed = 4
    seed2 = 5
    name = "random_gamma_5"
    input_dict = {"shape": shape, "alpha": alpha, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    shape = np.array([2, 1], dtype=np.int64)
    alpha = np.array([0.7, 0.8], dtype=np.float64).reshape((2,1))
    seed = 42
    seed2 = 24
    name = "random_gamma_6"
    input_dict = {"shape": shape, "alpha": alpha, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shape = np.array([1], dtype=np.int32)
    alpha = np.array([10.0], dtype=np.float32)
    seed = 1000
    seed2 = 2000
    name = "random_gamma_7"
    input_dict = {"shape": shape, "alpha": alpha, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = np.array([2, 2], dtype=np.int64)
    alpha = np.array([0.9, 1.1, 1.2, 1.4], dtype=np.float64).reshape((2,2))
    seed = 6
    seed2 = 7
    name = "random_gamma_8"
    input_dict = {"shape": shape, "alpha": alpha, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape = np.array([5, 5], dtype=np.int32)
    alpha = np.random.rand(25).astype(np.float32).reshape((5,5))
    seed = 8
    seed2 = 9
    name = "random_gamma_9"
    input_dict = {"shape": shape, "alpha": alpha, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = np.array([2, 3, 4], dtype=np.int64)
    alpha = np.random.rand(24).astype(np.float64).reshape((2,3,4))
    seed = 10
    seed2 = 11
    name = "random_gamma_10"
    input_dict = {"shape": shape, "alpha": alpha, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RandomGamma"] = tf_raw_ops_RandomGamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RandomGamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RandomGamma'.")

check_valid('tf.raw_ops.RandomGamma', generated_inputs['tf.raw_ops.RandomGamma'], lib="tf", suffix=0)
