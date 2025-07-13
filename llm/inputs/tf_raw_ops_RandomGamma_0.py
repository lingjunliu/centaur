
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_random_gamma_inputs():
    list_of_inputs = []

    # Input 1
    shape = np.array([2, 3], dtype=np.int32)
    alpha = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32).reshape((2,3))
    seed = 0
    seed2 = 0
    name = "gamma_1"
    input_dict = {"shape": tf.constant(shape), "alpha": tf.constant(alpha), "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = np.array([5], dtype=np.int64)
    alpha = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64)
    seed = 123
    seed2 = 456
    name = "gamma_2"
    input_dict = {"shape": tf.constant(shape), "alpha": tf.constant(alpha), "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = np.array([1, 4, 2], dtype=np.int32)
    alpha = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32).reshape((1,4,2))
    seed = 789
    seed2 = 101
    name = "gamma_3"
    input_dict = {"shape": tf.constant(shape), "alpha": tf.constant(alpha), "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = np.array([3, 2, 2], dtype=np.int64)
    alpha = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12], dtype=np.float64).reshape((3,2,2))
    seed = 0
    seed2 = 1
    name = "gamma_4"
    input_dict = {"shape": tf.constant(shape), "alpha": tf.constant(alpha), "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = np.array([4], dtype=np.int32)
    alpha = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    seed = -1
    seed2 = -1
    name = "gamma_5"
    input_dict = {"shape": tf.constant(shape), "alpha": tf.constant(alpha), "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = np.array([1], dtype=np.int64)
    alpha = np.array([10.0], dtype=np.float64)
    seed = 2**15 -1
    seed2 = 2**15 - 2
    name = "gamma_6"
    input_dict = {"shape": tf.constant(shape), "alpha": tf.constant(alpha), "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    shape = np.array([], dtype=np.int32)
    alpha = np.array(1.0, dtype=np.float32)
    seed = 0
    seed2 = 0
    name = "gamma_7"
    input_dict = {"shape": tf.constant(shape), "alpha": tf.constant(alpha), "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = np.array([2,2,2,2], dtype=np.int32)
    alpha = np.array([1]*16, dtype=np.float32).reshape((2,2,2,2))
    seed = 1
    seed2 = 0
    name = "gamma_8"
    input_dict = {"shape": tf.constant(shape), "alpha": tf.constant(alpha), "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    shape = np.array([1,1], dtype=np.int32)
    alpha = np.array([1.0], dtype=np.float32).reshape((1,1))
    seed = 0
    seed2 = 1
    name = "gamma_9"
    input_dict = {"shape": tf.constant(shape), "alpha": tf.constant(alpha), "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = np.array([2, 3], dtype=np.int64)
    alpha = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64).reshape((2,3))
    seed = 42
    seed2 = 1337
    name = "gamma_10"
    input_dict = {"shape": tf.constant(shape), "alpha": tf.constant(alpha), "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    shape = np.array([1], dtype=np.int32)
    alpha = np.array([1e-6], dtype=np.float32)
    seed = 0
    seed2 = 0
    name = "gamma_11"
    input_dict = {"shape": tf.constant(shape), "alpha": tf.constant(alpha), "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RandomGamma"] = tf_raw_ops_random_gamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RandomGamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RandomGamma'.")

check_valid('tf.raw_ops.RandomGamma', generated_inputs['tf.raw_ops.RandomGamma'], lib="tf", suffix=0)
