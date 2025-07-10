
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_gamma_inputs():
    list_of_inputs = []

    # Input 1
    shape = np.array([10], dtype=np.int32)
    alpha = np.array([0.5, 1.5], dtype=np.float32)
    beta = np.array([1.0, 2.0], dtype=np.float32)
    dtype = tf.float32
    seed = 123
    name = "gamma_samples_1"
    input_dict = {"shape": shape, "alpha": alpha, "beta": beta, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = np.array([7, 5], dtype=np.int32)
    alpha = np.array([0.5, 1.5], dtype=np.float32)
    beta = np.array([1.0], dtype=np.float32)
    dtype = tf.float32
    seed = 456
    name = "gamma_samples_2"
    input_dict = {"shape": shape, "alpha": alpha, "beta": beta, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = np.array([30], dtype=np.int32)
    alpha = np.array([[1.], [3.], [5.]], dtype=np.float32)
    beta = np.array([[3., 4.]], dtype=np.float32)
    dtype = tf.float32
    seed = 789
    name = "gamma_samples_3"
    input_dict = {"shape": shape, "alpha": alpha, "beta": beta, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = np.array([2, 3, 4], dtype=np.int32)
    alpha = np.array(1.0, dtype=np.float32)
    beta = np.array(2.0, dtype=np.float32)
    dtype = tf.float32
    seed = 101
    name = "gamma_samples_4"
    input_dict = {"shape": shape, "alpha": alpha, "beta": beta, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = np.array([5], dtype=np.int32)
    alpha = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    beta = np.array([2.0], dtype=np.float64)
    dtype = tf.float64
    seed = 202
    name = "gamma_samples_5"
    input_dict = {"shape": shape, "alpha": alpha, "beta": beta, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = np.array([1, 1], dtype=np.int32)
    alpha = np.array([[0.5]], dtype=np.float16)
    beta = np.array([[1.0]], dtype=np.float16)
    dtype = tf.float16
    seed = 303
    name = "gamma_samples_6"
    input_dict = {"shape": shape, "alpha": alpha, "beta": beta, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shape = np.array([10, 10], dtype=np.int32)
    alpha = np.array([1.0], dtype=np.float32)
    beta = np.array([0.1], dtype=np.float32)
    dtype = tf.float32
    seed = 404
    name = "gamma_samples_7"
    input_dict = {"shape": shape, "alpha": alpha, "beta": beta, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = np.array([1], dtype=np.int32)
    alpha = np.array([100.0], dtype=np.float32)
    beta = np.array([0.01], dtype=np.float32)
    dtype = tf.float32
    seed = 505
    name = "gamma_samples_8"
    input_dict = {"shape": shape, "alpha": alpha, "beta": beta, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    shape = np.array([2,2], dtype=np.int32)
    alpha = np.array([[2.0, 3.0],[4.0,5.0]], dtype=np.float32)
    beta = np.array([[0.5, 1.0],[1.5,2.0]], dtype=np.float32)
    dtype = tf.float32
    seed = 606
    name = "gamma_samples_9"
    input_dict = {"shape": shape, "alpha": alpha, "beta": beta, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = np.array([3], dtype=np.int32)
    alpha = np.array([0.7, 1.2, 2.1], dtype=np.float32)
    beta = np.array([0.3, 0.8, 1.5], dtype=np.float32)
    dtype = tf.float32
    seed = 707
    name = "gamma_samples_10"
    input_dict = {"shape": shape, "alpha": alpha, "beta": beta, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.gamma"] = tf_random_gamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random.gamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.gamma'.")

check_valid('tf.random.gamma', generated_inputs['tf.random.gamma'], lib="tf", suffix=0)
