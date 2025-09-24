
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_gamma_inputs():
    list_of_inputs = []

    # Input 1
    shape = np.array([10, 2], dtype=np.int32)
    seed = np.array([12, 34], dtype=np.int32)
    alpha = np.array([0.5, 1.5], dtype=np.float32)
    beta = np.array([1.0, 2.0], dtype=np.float32)
    dtype = tf.float32

    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = np.array([7, 5, 2], dtype=np.int32)
    seed = np.array([12, 34], dtype=np.int32)
    alpha = np.array([0.5, 1.5], dtype=np.float32)
    beta = np.array([1.0, 2.0], dtype=np.float32)
    dtype = tf.float32

    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = np.array([30, 3, 2], dtype=np.int32)
    seed = np.array([12, 34], dtype=np.int32)
    alpha = np.array([[1.], [3.], [5.]], dtype=np.float32)
    beta = np.array([[3., 4.]], dtype=np.float32)
    dtype = tf.float32

    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = np.array([5, 5], dtype=np.int32)
    seed = np.array([100, 200], dtype=np.int32)
    alpha = np.array(2.5, dtype=np.float32)
    beta = np.array(0.5, dtype=np.float32)
    dtype = tf.float32

    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = np.array([2, 3, 4], dtype=np.int32)
    seed = np.array([50, 60], dtype=np.int32)
    alpha = np.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
                     [[13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]]], dtype=np.float32)
    beta = np.array(1.0, dtype=np.float32)
    dtype = tf.float32

    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = np.array([4, 2], dtype=np.int32)
    seed = np.array([70, 80], dtype=np.int32)
    alpha = np.array([0.1, 0.2], dtype=np.float32)
    beta = np.array([10.0, 20.0], dtype=np.float32)
    dtype = tf.float32

    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shape = np.array([2, 2, 2], dtype=np.int32)
    seed = np.array([1, 2], dtype=np.int32)
    alpha = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    beta = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    dtype = tf.float32

    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = np.array([2, 2, 2], dtype=np.int32)
    seed = np.array([1, 2], dtype=np.int32)
    alpha = np.array(2.0, dtype=np.float32)
    beta = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    dtype = tf.float32

    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9
    shape = np.array([5], dtype=np.int32)
    seed = np.array([10, 20], dtype=np.int32)
    alpha = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float32)
    beta = np.array(1.0, dtype=np.float32)
    dtype = tf.float32

    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = np.array([2, 3], dtype=np.int32)
    seed = np.array([10, 20], dtype=np.int32)
    alpha = np.array([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]], dtype=np.float32)
    beta = np.array(0.1, dtype=np.float32)
    dtype = tf.float32

    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    shape = np.array([2, 3], dtype=np.int32)
    seed = np.array([10, 20], dtype=np.int32)
    alpha = np.array([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]], dtype=np.float64)
    beta = np.array(0.1, dtype=np.float64)
    dtype = tf.float64

    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.stateless_gamma"] = tf_random_stateless_gamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random.stateless_gamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_gamma'.")

check_valid('tf.random.stateless_gamma', generated_inputs['tf.random.stateless_gamma'], lib="tf", suffix=0)
