
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_fractional_avg_pool_inputs():
    list_of_inputs = []

    # Input 1
    value = np.random.rand(1, 10, 10, 1).astype(np.float32)
    pooling_ratio = [1.0, 1.44, 1.73, 1.0]
    pseudo_random = False
    overlapping = False
    seed = 0
    name = "fractional_avg_pool_1"
    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    value = np.random.rand(2, 20, 30, 3).astype(np.float32)
    pooling_ratio = [1.0, 1.2, 1.5, 1.0]
    pseudo_random = True
    overlapping = True
    seed = 123
    name = "fractional_avg_pool_2"
    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    value = np.random.rand(1, 5, 7, 1).astype(np.float32)
    pooling_ratio = [1.0, 1.1, 1.3, 1.0]
    pseudo_random = False
    overlapping = True
    seed = 42
    name = "fractional_avg_pool_3"
    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    value = np.random.rand(4, 15, 25, 5).astype(np.float32)
    pooling_ratio = [1.0, 1.6, 1.8, 1.0]
    pseudo_random = True
    overlapping = False
    seed = 99
    name = "fractional_avg_pool_4"
    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    value = np.random.rand(1, 8, 12, 1).astype(np.float32)
    pooling_ratio = [1.0, 1.3, 1.6, 1.0]
    pseudo_random = False
    overlapping = False
    seed = 0
    name = "fractional_avg_pool_5"
    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    value = np.random.rand(2, 16, 24, 2).astype(np.float32)
    pooling_ratio = [1.0, 1.5, 1.7, 1.0]
    pseudo_random = True
    overlapping = True
    seed = 55
    name = "fractional_avg_pool_6"
    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    value = np.random.rand(1, 4, 6, 1).astype(np.float32)
    pooling_ratio = [1.0, 1.2, 1.4, 1.0]
    pseudo_random = False
    overlapping = True
    seed = 11
    name = "fractional_avg_pool_7"
    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    value = np.random.rand(3, 12, 18, 4).astype(np.float32)
    pooling_ratio = [1.0, 1.7, 1.9, 1.0]
    pseudo_random = True
    overlapping = False
    seed = 77
    name = "fractional_avg_pool_8"
    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    value = np.random.rand(1, 6, 9, 1).astype(np.float32)
    pooling_ratio = [1.0, 1.15, 1.25, 1.0]
    pseudo_random = False
    overlapping = False
    seed = 0
    name = "fractional_avg_pool_9"
    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    value = np.random.rand(2, 14, 21, 2).astype(np.float32)
    pooling_ratio = [1.0, 1.55, 1.65, 1.0]
    pseudo_random = True
    overlapping = True
    seed = 33
    name = "fractional_avg_pool_10"
    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    value = np.random.rand(1, 7, 11, 1).astype(np.float32)
    pooling_ratio = [1.0, 1.35, 1.45, 1.0]
    pseudo_random = False
    overlapping = True
    seed = 88
    name = "fractional_avg_pool_11"
    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    value = np.random.rand(3, 13, 19, 4).astype(np.float32)
    pooling_ratio = [1.0, 1.8, 1.95, 1.0]
    pseudo_random = True
    overlapping = False
    seed = 22
    name = "fractional_avg_pool_12"
    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.fractional_avg_pool"] = tf_nn_fractional_avg_pool_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.fractional_avg_pool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.fractional_avg_pool'.")

check_valid('tf.nn.fractional_avg_pool', generated_inputs['tf.nn.fractional_avg_pool'], lib="tf", suffix=0)
