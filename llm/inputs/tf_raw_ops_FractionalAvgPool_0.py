
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FractionalAvgPool_inputs():
    list_of_inputs = []

    # Input 1
    value = np.random.rand(1, 10, 10, 1).astype(np.float32)
    pooling_ratio = [1.0, 1.44, 1.73, 1.0]
    pseudo_random = False
    overlapping = False
    deterministic = False
    seed = 0
    seed2 = 0
    name = "test_pool1"

    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "deterministic": deterministic,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    value = np.random.rand(2, 20, 20, 3).astype(np.float64)
    pooling_ratio = [1.0, 1.2, 1.5, 1.0]
    pseudo_random = True
    overlapping = True
    deterministic = True
    seed = 123
    seed2 = 456
    name = "test_pool2"

    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "deterministic": deterministic,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    value = np.random.rand(4, 5, 5, 2).astype(np.int32)
    pooling_ratio = [1.0, 1.1, 1.3, 1.0]
    pseudo_random = False
    overlapping = True
    deterministic = False
    seed = 789
    seed2 = 101
    name = "test_pool3"

    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "deterministic": deterministic,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    value = np.random.rand(1, 15, 15, 1).astype(np.int64)
    pooling_ratio = [1.0, 2.0, 2.0, 1.0]
    pseudo_random = True
    overlapping = False
    deterministic = True
    seed = 0
    seed2 = 1
    name = "test_pool4"

    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "deterministic": deterministic,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    value = np.random.rand(3, 8, 8, 4).astype(np.float32)
    pooling_ratio = [1.0, 1.6, 1.8, 1.0]
    pseudo_random = False
    overlapping = True
    deterministic = False
    seed = 1
    seed2 = 0
    name = "test_pool5"

    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "deterministic": deterministic,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    value = np.random.rand(1, 32, 32, 1).astype(np.float64)
    pooling_ratio = [1.0, 1.1, 1.2, 1.0]
    pseudo_random = True
    overlapping = False
    deterministic = True
    seed = 2
    seed2 = 2
    name = "test_pool6"

    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "deterministic": deterministic,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    value = np.random.rand(2, 7, 7, 2).astype(np.int32)
    pooling_ratio = [1.0, 1.7, 1.9, 1.0]
    pseudo_random = False
    overlapping = True
    deterministic = False
    seed = 3
    seed2 = 3
    name = "test_pool7"

    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "deterministic": deterministic,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    value = np.random.rand(4, 12, 12, 4).astype(np.int64)
    pooling_ratio = [1.0, 1.3, 1.4, 1.0]
    pseudo_random = True
    overlapping = False
    deterministic = True
    seed = 4
    seed2 = 4
    name = "test_pool8"

    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "deterministic": deterministic,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    value = np.random.rand(1, 4, 4, 1).astype(np.float32)
    pooling_ratio = [1.0, 1.9, 1.1, 1.0]
    pseudo_random = False
    overlapping = True
    deterministic = False
    seed = 5
    seed2 = 5
    name = "test_pool9"

    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "deterministic": deterministic,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    value = np.random.rand(2, 16, 16, 3).astype(np.float64)
    pooling_ratio = [1.0, 1.25, 1.75, 1.0]
    pseudo_random = True
    overlapping = False
    deterministic = True
    seed = 6
    seed2 = 6
    name = "test_pool10"

    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "deterministic": deterministic,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.FractionalAvgPool"] = tf_raw_ops_FractionalAvgPool_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FractionalAvgPool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FractionalAvgPool'.")

check_valid('tf.raw_ops.FractionalAvgPool', generated_inputs['tf.raw_ops.FractionalAvgPool'], lib="tf", suffix=0)
