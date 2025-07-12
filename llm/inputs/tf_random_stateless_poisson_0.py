
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_poisson_inputs():
    list_of_inputs = []

    # Input 1
    shape = np.array([10, 2], dtype=np.int32)
    seed = np.array([12, 34], dtype=np.int32)
    lam = np.array([5, 15], dtype=np.float32)
    dtype = tf.int32
    name = "poisson_sample_1"
    input_dict = {"shape": shape, "seed": seed, "lam": lam, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = np.array([7, 5, 2], dtype=np.int32)
    seed = np.array([12, 34], dtype=np.int32)
    lam = np.array([5, 15], dtype=np.float32)
    dtype = tf.int64
    name = "poisson_sample_2"
    input_dict = {"shape": shape, "seed": seed, "lam": lam, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = np.array([30, 3, 1], dtype=np.int32)
    seed = np.array([12, 34], dtype=np.int32)
    lam = np.array([[1.], [3.], [5.]], dtype=np.float32)
    dtype = tf.int32
    name = "poisson_sample_3"
    input_dict = {"shape": shape, "seed": seed, "lam": lam, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = np.array([5, 4, 2], dtype=np.int32)
    seed = np.array([100, 200], dtype=np.int32)
    lam = np.array([1.0, 2.0], dtype=np.float32)
    dtype = tf.int32
    name = "poisson_sample_4"
    input_dict = {"shape": shape, "seed": seed, "lam": lam, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = np.array([2, 2], dtype=np.int32)
    seed = np.array([1, 2], dtype=np.int32)
    lam = np.array([0.5, 1.5], dtype=np.float32)
    dtype = tf.int32
    name = "poisson_sample_5"
    input_dict = {"shape": shape, "seed": seed, "lam": lam, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = np.array([1, 5], dtype=np.int32)
    seed = np.array([5, 6], dtype=np.int32)
    lam = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    dtype = tf.int32
    name = "poisson_sample_6"
    input_dict = {"shape": shape, "seed": seed, "lam": lam, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shape = np.array([4, 1], dtype=np.int32)
    seed = np.array([7, 8], dtype=np.int32)
    lam = np.array([7.0], dtype=np.float32)
    dtype = tf.int32
    name = "poisson_sample_7"
    input_dict = {"shape": shape, "seed": seed, "lam": lam, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = np.array([2, 3, 4], dtype=np.int32)
    seed = np.array([9, 10], dtype=np.int32)
    lam = np.array([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]]], dtype=np.float32)
    dtype = tf.int32
    name = "poisson_sample_8"
    input_dict = {"shape": shape, "seed": seed, "lam": lam, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape = np.array([5], dtype=np.int32)
    seed = np.array([11, 12], dtype=np.int32)
    lam = np.array([1.0], dtype=np.float32)
    dtype = tf.int32
    name = "poisson_sample_9"
    input_dict = {"shape": shape, "seed": seed, "lam": lam, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = np.array([2, 2], dtype=np.int32)
    seed = np.array([13, 14], dtype=np.int32)
    lam = np.array([1.0, 2.0], dtype=np.float64)
    dtype = tf.int32
    name = "poisson_sample_10"
    input_dict = {"shape": shape, "seed": seed, "lam": lam, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11 - Adjusted to avoid broadcasting issues
    shape = np.array([4, 2], dtype=np.int32)
    seed = np.array([15, 16], dtype=np.int32)
    lam = np.array([0.2, 1.3], dtype=np.float32)
    dtype = tf.int32
    name = "poisson_sample_11"
    input_dict = {"shape": shape, "seed": seed, "lam": lam, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    shape = np.array([2,1], dtype=np.int32)
    seed = np.array([17, 18], dtype=np.int32)
    lam = np.array([10.0], dtype=np.float32)
    dtype = tf.int32
    name = "poisson_sample_12"
    input_dict = {"shape": shape, "seed": seed, "lam": lam, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13 - Matching shapes
    shape = np.array([3, 2, 2], dtype=np.int32)
    seed = np.array([20, 21], dtype=np.int32)
    lam = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    dtype = tf.int32
    name = "poisson_sample_13"
    input_dict = {"shape": shape, "seed": seed, "lam": lam, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.stateless_poisson"] = tf_random_stateless_poisson_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random.stateless_poisson' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_poisson'.")

check_valid('tf.random.stateless_poisson', generated_inputs['tf.random.stateless_poisson'], lib="tf", suffix=0)
