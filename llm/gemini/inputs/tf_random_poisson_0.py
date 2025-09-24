
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_poisson_inputs():
    list_of_inputs = []

    # Input 1
    shape = np.array([10], dtype=np.int32)
    lam = np.array([0.5, 1.5], dtype=np.float32)
    dtype = tf.float32
    seed = 123
    name = "poisson_samples_1"
    input_dict = {"shape": shape, "lam": lam, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = np.array([7, 5], dtype=np.int32)
    lam = np.array([12.2, 3.3], dtype=np.float32)
    dtype = tf.float32
    seed = 456
    name = "poisson_samples_2"
    input_dict = {"shape": shape, "lam": lam, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = np.array([2, 3, 4], dtype=np.int32)
    lam = np.array([5.0], dtype=np.float32)
    dtype = tf.float64
    seed = 789
    name = "poisson_samples_3"
    input_dict = {"shape": shape, "lam": lam, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = np.array([100], dtype=np.int32)
    lam = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    dtype = tf.int32
    seed = 101
    name = "poisson_samples_4"
    input_dict = {"shape": shape, "lam": lam, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = np.array([5, 5], dtype=np.int32)
    lam = np.array([1.0], dtype=np.float32)
    dtype = tf.int64
    seed = 202
    name = "poisson_samples_5"
    input_dict = {"shape": shape, "lam": lam, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = np.array([1], dtype=np.int32)
    lam = np.array([100.0], dtype=np.float32)
    dtype = tf.float16
    seed = 303
    name = "poisson_samples_6"
    input_dict = {"shape": shape, "lam": lam, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shape = np.array([2, 2, 2, 2], dtype=np.int32)
    lam = np.array([2.5], dtype=np.float32)
    dtype = tf.float32
    seed = 404
    name = "poisson_samples_7"
    input_dict = {"shape": shape, "lam": lam, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = np.array([3, 1], dtype=np.int32)
    lam = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    dtype = tf.float32
    seed = 505
    name = "poisson_samples_8"
    input_dict = {"shape": shape, "lam": lam, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape = np.array([4], dtype=np.int32)
    lam = np.array([0.7], dtype=np.float32)
    dtype = tf.int32
    seed = 606
    name = "poisson_samples_9"
    input_dict = {"shape": shape, "lam": lam, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = np.array([2, 3], dtype=np.int32)
    lam = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    dtype = tf.float64
    seed = 707
    name = "poisson_samples_10"
    input_dict = {"shape": shape, "lam": lam, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    shape = np.array([1, 1], dtype=np.int32)
    lam = np.array(1.0, dtype=np.float32)
    dtype = tf.int64
    seed = 808
    name = "poisson_samples_11"
    input_dict = {"shape": shape, "lam": lam, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.poisson"] = tf_random_poisson_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random.poisson' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.poisson'.")

check_valid('tf.random.poisson', generated_inputs['tf.random.poisson'], lib="tf", suffix=0)
