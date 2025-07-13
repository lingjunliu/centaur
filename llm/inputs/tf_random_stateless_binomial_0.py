
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_binomial_inputs():
    list_of_inputs = []

    # Input 1
    shape = np.array([5], dtype=np.int32)
    seed = np.array([123, 456], dtype=np.int32)
    counts = np.array([10], dtype=np.float32)
    probs = np.array([0.5], dtype=np.float32)
    output_dtype = tf.int32
    name = "binomial_sample_1"

    input_dict = {
        "shape": shape,
        "seed": seed,
        "counts": counts,
        "probs": probs,
        "output_dtype": output_dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = np.array([2, 3], dtype=np.int32)
    seed = np.array([789, 101], dtype=np.int32)
    counts = np.array([10, 10, 10], dtype=np.float32)
    probs = np.array([0.4, 0.4, 0.4], dtype=np.float32)
    output_dtype = tf.int64
    name = "binomial_sample_2"

    input_dict = {
        "shape": shape,
        "seed": seed,
        "counts": counts,
        "probs": probs,
        "output_dtype": output_dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = np.array([4, 2], dtype=np.int32)
    seed = np.array([222, 333], dtype=np.int32)
    counts = np.array([35, 35, 35, 35], dtype=np.float32).reshape((4,1))
    probs = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32).reshape((4,1))
    output_dtype = tf.int32
    name = "binomial_sample_3"

    input_dict = {
        "shape": shape,
        "seed": seed,
        "counts": counts,
        "probs": probs,
        "output_dtype": output_dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = np.array([10], dtype=np.int32)
    seed = np.array([444, 555], dtype=np.int32)
    counts = np.array([50], dtype=np.float32)
    probs = np.array([0.95], dtype=np.float32)
    output_dtype = tf.int32
    name = "binomial_sample_4"

    input_dict = {
        "shape": shape,
        "seed": seed,
        "counts": counts,
        "probs": probs,
        "output_dtype": output_dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = np.array([3, 3, 3], dtype=np.int32)
    seed = np.array([666, 777], dtype=np.int32)
    counts = np.array([10], dtype=np.float32)
    probs = np.array([0.5], dtype=np.float32)
    output_dtype = tf.int64
    name = "binomial_sample_5"

    input_dict = {
        "shape": shape,
        "seed": seed,
        "counts": counts,
        "probs": probs,
        "output_dtype": output_dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = np.array([7], dtype=np.int32)
    seed = np.array([888, 999], dtype=np.int32)
    counts = np.array([100], dtype=np.float32)
    probs = np.array([0.05], dtype=np.float32)
    output_dtype = tf.int32
    name = "binomial_sample_6"

    input_dict = {
        "shape": shape,
        "seed": seed,
        "counts": counts,
        "probs": probs,
        "output_dtype": output_dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shape = np.array([2, 5], dtype=np.int32)
    seed = np.array([1000, 1001], dtype=np.int32)
    counts = np.array([50, 50, 50, 50, 50], dtype=np.float32)
    probs = np.array([0.6, 0.6, 0.6, 0.6, 0.6], dtype=np.float32)
    output_dtype = tf.int64
    name = "binomial_sample_7"

    input_dict = {
        "shape": shape,
        "seed": seed,
        "counts": counts,
        "probs": probs,
        "output_dtype": output_dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = np.array([1, 1, 1], dtype=np.int32)
    seed = np.array([1002, 1003], dtype=np.int32)
    counts = np.array([100.], dtype=np.float32)
    probs = np.array([0.9], dtype=np.float32)
    output_dtype = tf.int32
    name = "binomial_sample_8"

    input_dict = {
        "shape": shape,
        "seed": seed,
        "counts": counts,
        "probs": probs,
        "output_dtype": output_dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape = np.array([2, 2, 2, 2], dtype=np.int32)
    seed = np.array([1004, 1005], dtype=np.int32)
    counts = np.array([20], dtype=np.float32)
    probs = np.array([0.6], dtype=np.float32)

    output_dtype = tf.int64
    name = "binomial_sample_9"

    input_dict = {
        "shape": shape,
        "seed": seed,
        "counts": counts,
        "probs": probs,
        "output_dtype": output_dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = np.array([4], dtype=np.int32)
    seed = np.array([1006, 1007], dtype=np.int32)
    counts = np.array([10, 11, 12, 13], dtype=np.float32)
    probs = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    output_dtype = tf.int32
    name = "binomial_sample_10"

    input_dict = {
        "shape": shape,
        "seed": seed,
        "counts": counts,
        "probs": probs,
        "output_dtype": output_dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.stateless_binomial"] = tf_random_stateless_binomial_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random.stateless_binomial' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_binomial'.")

check_valid('tf.random.stateless_binomial', generated_inputs['tf.random.stateless_binomial'], lib="tf", suffix=0)
