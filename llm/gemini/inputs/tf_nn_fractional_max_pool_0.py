
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_fractional_max_pool_inputs():
    list_of_inputs = []

    # Input 1
    value = np.random.rand(1, 4, 4, 1).astype(np.float32)
    pooling_ratio = [1.0, 1.44, 1.73, 1.0]
    pseudo_random = False
    overlapping = False
    seed = 1
    name = "fractional_max_pool_1"
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": pseudo_random, "overlapping": overlapping, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    value = np.random.rand(2, 8, 8, 3).astype(np.float32)
    pooling_ratio = [1.0, 1.1, 1.2, 1.0]
    pseudo_random = True
    overlapping = True
    seed = 123
    name = "fractional_max_pool_2"
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": pseudo_random, "overlapping": overlapping, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    value = np.random.rand(1, 16, 16, 1).astype(np.float32)
    pooling_ratio = [1.0, 1.7, 1.3, 1.0]
    pseudo_random = False
    overlapping = True
    seed = 456
    name = "fractional_max_pool_3"
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": pseudo_random, "overlapping": overlapping, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    value = np.random.rand(4, 4, 4, 4).astype(np.float32)
    pooling_ratio = [1.0, 1.2, 1.5, 1.0]
    pseudo_random = True
    overlapping = False
    seed = 789
    name = "fractional_max_pool_4"
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": pseudo_random, "overlapping": overlapping, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    value = np.random.rand(1, 32, 32, 1).astype(np.float32)
    pooling_ratio = [1.0, 2.0, 1.0, 1.0]
    pseudo_random = False
    overlapping = True
    seed = 101
    name = "fractional_max_pool_5"
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": pseudo_random, "overlapping": overlapping, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    value = np.random.rand(2, 64, 64, 3).astype(np.float32)
    pooling_ratio = [1.0, 1.0, 2.0, 1.0]
    pseudo_random = True
    overlapping = False
    seed = 202
    name = "fractional_max_pool_6"
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": pseudo_random, "overlapping": overlapping, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    value = np.random.rand(1, 128, 128, 1).astype(np.float32)
    pooling_ratio = [1.0, 1.5, 1.5, 1.0]
    pseudo_random = False
    overlapping = False
    seed = 303
    name = "fractional_max_pool_7"
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": pseudo_random, "overlapping": overlapping, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    value = np.random.rand(4, 2, 2, 4).astype(np.float32)
    pooling_ratio = [1.0, 1.9, 1.1, 1.0]
    pseudo_random = True
    overlapping = True
    seed = 404
    name = "fractional_max_pool_8"
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": pseudo_random, "overlapping": overlapping, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    value = np.random.rand(1, 5, 5, 1).astype(np.float32)
    pooling_ratio = [1.0, 1.3, 1.8, 1.0]
    pseudo_random = False
    overlapping = True
    seed = 505
    name = "fractional_max_pool_9"
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": pseudo_random, "overlapping": overlapping, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    value = np.random.rand(2, 10, 10, 3).astype(np.float32)
    pooling_ratio = [1.0, 1.6, 1.4, 1.0]
    pseudo_random = True
    overlapping = False
    seed = 606
    name = "fractional_max_pool_10"
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": pseudo_random, "overlapping": overlapping, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.fractional_max_pool"] = tf_nn_fractional_max_pool_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.fractional_max_pool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.fractional_max_pool'.")

check_valid('tf.nn.fractional_max_pool', generated_inputs['tf.nn.fractional_max_pool'], lib="tf", suffix=0)
