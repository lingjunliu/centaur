
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_fractional_avg_pool_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "value": np.random.rand(1, 10, 10, 3).astype(np.float32),
        "pooling_ratio": [1.0, 1.5, 1.5, 1.0],
        "pseudo_random": False,
        "overlapping": False,
        "seed": 42,
        "name": "avg_pool_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "value": np.random.rand(2, 20, 20, 1).astype(np.float64),
        "pooling_ratio": [1.0, 2.0, 2.0, 1.0],
        "pseudo_random": True,
        "overlapping": True,
        "seed": 10,
        "name": "avg_pool_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "value": np.ones((1, 8, 8, 16), dtype=np.float32),
        "pooling_ratio": [1.0, 1.2, 1.3, 1.0],
        "pseudo_random": False,
        "overlapping": True,
        "seed": 0,
        "name": "avg_pool_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "value": np.random.randn(4, 15, 15, 3).astype(np.float32),
        "pooling_ratio": [1.0, 1.1, 1.1, 1.0],
        "pseudo_random": True,
        "overlapping": False,
        "seed": 123,
        "name": "avg_pool_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "value": np.zeros((1, 100, 100, 3), dtype=np.float32),
        "pooling_ratio": [1.0, 1.0, 1.0, 1.0],
        "pseudo_random": False,
        "overlapping": False,
        "seed": 99,
        "name": "avg_pool_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "value": np.random.rand(1, 5, 5, 2).astype(np.float32),
        "pooling_ratio": [1.0, 1.8, 1.8, 1.0],
        "pseudo_random": True,
        "overlapping": True,
        "seed": 5,
        "name": "avg_pool_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "value": (np.random.rand(2, 12, 12, 4) * 10).astype(np.float32),
        "pooling_ratio": [1.0, 1.44, 1.73, 1.0],
        "pseudo_random": False,
        "overlapping": False,
        "seed": 7,
        "name": "avg_pool_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "value": np.random.randn(3, 30, 30, 8).astype(np.float32),
        "pooling_ratio": [1.0, 2.5, 2.5, 1.0],
        "pseudo_random": True,
        "overlapping": False,
        "seed": 1001,
        "name": "avg_pool_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "value": np.arange(144, dtype=np.float32).reshape((1, 12, 12, 1)),
        "pooling_ratio": [1.0, 1.5, 1.2, 1.0],
        "pseudo_random": False,
        "overlapping": True,
        "seed": 88,
        "name": "avg_pool_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "value": np.random.uniform(-1, 1, (2, 16, 16, 3)).astype(np.float32),
        "pooling_ratio": [1.0, 1.33, 1.33, 1.0],
        "pseudo_random": True,
        "overlapping": True,
        "seed": 55,
        "name": "avg_pool_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.fractional_avg_pool"] = tf_nn_fractional_avg_pool_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.fractional_avg_pool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.fractional_avg_pool'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.fractional_avg_pool', generated_inputs['tf.nn.fractional_avg_pool'], lib="tf", suffix=0)
