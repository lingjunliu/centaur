
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
    name = "fractional_avg_pool_1"
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": pseudo_random, "overlapping": overlapping, "deterministic": deterministic, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    value = np.random.rand(2, 20, 20, 3).astype(np.float64)
    pooling_ratio = [1.0, 1.2, 1.5, 1.0]
    pseudo_random = True
    overlapping = True
    deterministic = True
    seed = 1
    seed2 = 2
    name = "fractional_avg_pool_2"
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": pseudo_random, "overlapping": overlapping, "deterministic": deterministic, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    value = np.random.randint(0, 10, size=(4, 15, 15, 2), dtype=np.int32)
    pooling_ratio = [1.0, 1.8, 2.1, 1.0]
    pseudo_random = False
    overlapping = True
    deterministic = False
    seed = 123
    seed2 = 456
    name = "fractional_avg_pool_3"
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": pseudo_random, "overlapping": overlapping, "deterministic": deterministic, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    value = np.random.randint(-10, 10, size=(1, 5, 5, 1), dtype=np.int64)
    pooling_ratio = [1.0, 1.1, 1.3, 1.0]
    pseudo_random = True
    overlapping = False
    deterministic = True
    seed = 789
    seed2 = 101
    name = "fractional_avg_pool_4"
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": pseudo_random, "overlapping": overlapping, "deterministic": deterministic, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    value = np.random.rand(1, 8, 8, 1).astype(np.float32)
    pooling_ratio = [1.0, 1.99, 1.99, 1.0]
    pseudo_random = False
    overlapping = False
    deterministic = False
    seed = 0
    seed2 = 0
    name = None
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": pseudo_random, "overlapping": overlapping, "deterministic": deterministic, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    value = np.random.rand(3, 16, 16, 4).astype(np.float64)
    pooling_ratio = [1.0, 1.6, 1.8, 1.0]
    pseudo_random = True
    overlapping = True
    deterministic = True
    seed = 42
    seed2 = 24
    name = "fractional_avg_pool_6"
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": pseudo_random, "overlapping": overlapping, "deterministic": deterministic, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    value = np.random.randint(0, 5, size=(2, 7, 7, 1), dtype=np.int32)
    pooling_ratio = [1.0, 1.3, 1.4, 1.0]
    pseudo_random = False
    overlapping = True
    deterministic = False
    seed = 99
    seed2 = 11
    name = "fractional_avg_pool_7"
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": pseudo_random, "overlapping": overlapping, "deterministic": deterministic, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    value = np.random.randint(-5, 5, size=(1, 12, 12, 1), dtype=np.int64)
    pooling_ratio = [1.0, 1.7, 1.9, 1.0]
    pseudo_random = True
    overlapping = False
    deterministic = True
    seed = 55
    seed2 = 66
    name = "fractional_avg_pool_8"
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": pseudo_random, "overlapping": overlapping, "deterministic": deterministic, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    value = np.random.rand(1, 6, 6, 1).astype(np.float32)
    pooling_ratio = [1.0, 1.01, 1.02, 1.0]
    pseudo_random = False
    overlapping = False
    deterministic = False
    seed = 0
    seed2 = 0
    name = "fractional_avg_pool_9"
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": pseudo_random, "overlapping": overlapping, "deterministic": deterministic, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    value = np.random.rand(4, 32, 32, 8).astype(np.float64)
    pooling_ratio = [1.0, 1.35, 1.65, 1.0]
    pseudo_random = True
    overlapping = True
    deterministic = True
    seed = 88
    seed2 = 99
    name = "fractional_avg_pool_10"
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": pseudo_random, "overlapping": overlapping, "deterministic": deterministic, "seed": seed, "seed2": seed2, "name": name}
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
