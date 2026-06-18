
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_avg_pool1d_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "input": np.random.randn(2, 10, 3).astype(np.float32),
        "ksize": [2],
        "strides": [2],
        "padding": "VALID",
        "data_format": "NWC",
        "name": "pool_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "input": np.ones((2, 8, 3), dtype=np.float32),
        "ksize": [3],
        "strides": [2],
        "padding": "SAME",
        "data_format": "NWC",
        "name": "pool_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "input": np.zeros((1, 10, 4), dtype=np.float32),
        "ksize": [1, 2, 1],
        "strides": [1, 1, 1],
        "padding": "VALID",
        "data_format": "NWC",
        "name": "pool_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "input": np.random.uniform(-5.0, 5.0, (3, 6, 4)).astype(np.float32),
        "ksize": [1, 3, 1],
        "strides": [1, 2, 1],
        "padding": "SAME",
        "data_format": "NWC",
        "name": "pool_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "input": np.arange(30, dtype=np.float32).reshape(1, 10, 3),
        "ksize": [4],
        "strides": [3],
        "padding": "VALID",
        "data_format": "NWC",
        "name": "pool_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "input": np.random.randn(2, 12, 4).astype(np.float64),
        "ksize": [2],
        "strides": [1],
        "padding": "SAME",
        "data_format": "NWC",
        "name": "pool_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "input": np.ones((1, 15, 1), dtype=np.float32) * -2.5,
        "ksize": [1, 5, 1],
        "strides": [1, 5, 1],
        "padding": "VALID",
        "data_format": "NWC",
        "name": "pool_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "input": np.random.randn(4, 16, 3).astype(np.float32),
        "ksize": [1, 2, 1],
        "strides": [1, 2, 1],
        "padding": "SAME",
        "data_format": "NWC",
        "name": "pool_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "input": np.arange(24, dtype=np.float32).reshape(2, 4, 3),
        "ksize": [3],
        "strides": [1],
        "padding": "SAME",
        "data_format": "NWC",
        "name": "pool_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "input": np.random.normal(size=(1, 5, 5)).astype(np.float32),
        "ksize": [3],
        "strides": [2],
        "padding": "VALID",
        "data_format": "NWC",
        "name": "pool_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.avg_pool1d"] = tf_nn_avg_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.avg_pool1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.avg_pool1d'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.avg_pool1d', generated_inputs['tf.nn.avg_pool1d'], lib="tf", suffix=0)
