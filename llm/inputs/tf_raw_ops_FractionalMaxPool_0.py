
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FractionalMaxPool_inputs():
    list_of_inputs = []

    # Input 1
    value = np.random.rand(1, 10, 10, 1).astype(np.float32)
    pooling_ratio = [1.0, 1.44, 1.73, 1.0]
    pseudo_random = False
    overlapping = False
    deterministic = False
    seed = 0
    seed2 = 0
    name = None
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": pseudo_random, "overlapping": overlapping, "deterministic": deterministic, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    value = np.random.rand(4, 20, 20, 3).astype(np.float64)
    pooling_ratio = [1.0, 1.2, 1.5, 1.0]
    pseudo_random = True
    overlapping = True
    deterministic = True
    seed = 123
    seed2 = 456
    name = "fractional_max_pool_1"
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": pseudo_random, "overlapping": overlapping, "deterministic": deterministic, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_FractionalMaxPool_inputs()
generated_inputs["tf.raw_ops.FractionalMaxPool"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FractionalMaxPool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FractionalMaxPool'.")

check_valid('tf.raw_ops.FractionalMaxPool', generated_inputs['tf.raw_ops.FractionalMaxPool'], lib="tf", suffix=0)
