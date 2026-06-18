
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool2d_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "input": np.random.randn(1, 4, 4, 1).astype(np.float32),
        "ksize": 2,
        "strides": [2, 2],
        "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "data_format": "NHWC",
        "name": "pool1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "input": np.random.randn(2, 6, 6, 3).astype(np.float32),
        "ksize": 3,
        "strides": [1, 1],
        "padding": [[0, 0], [1, 1], [1, 1], [0, 0]],
        "data_format": "NHWC",
        "name": "pool2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "input": np.random.randn(1, 8, 8, 3).astype(np.float32),
        "ksize": 2,
        "strides": [2, 2],
        "padding": [[0, 0], [1, 1], [1, 1], [0, 0]],
        "data_format": "NHWC",
        "name": "pool3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "input": np.random.randn(1, 4, 4, 2).astype(np.float32),
        "ksize": 2,
        "strides": [1, 1, 1, 1],
        "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "data_format": "NHWC",
        "name": "pool4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "input": np.random.randn(1, 5, 5, 2).astype(np.float64),
        "ksize": 2,
        "strides": [1, 2, 2, 1],
        "padding": [[0, 0], [1, 1], [1, 1], [0, 0]],
        "data_format": "NHWC",
        "name": "pool5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "input": np.random.randn(1, 4, 4, 1).astype(np.float16),
        "ksize": 2,
        "strides": [1, 1],
        "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "data_format": "NHWC",
        "name": "pool6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "input": -np.abs(np.random.randn(2, 5, 5, 2).astype(np.float32)),
        "ksize": 2,
        "strides": [2, 2],
        "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "data_format": "NHWC",
        "name": "pool7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "input": np.random.randn(1, 10, 10, 3).astype(np.float32),
        "ksize": 3,
        "strides": [2, 2],
        "padding": [[0, 0], [1, 1], [1, 1], [0, 0]],
        "data_format": "NHWC",
        "name": "pool8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "input": np.random.randn(2, 2, 2, 2).astype(np.float32),
        "ksize": 1,
        "strides": [1, 1],
        "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "data_format": "NHWC",
        "name": "pool9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "input": np.random.randn(3, 10, 10, 4).astype(np.float32),
        "ksize": 4,
        "strides": [3, 3],
        "padding": [[0, 0], [1, 2], [2, 1], [0, 0]],
        "data_format": "NHWC",
        "name": "pool10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.max_pool2d_7"] = tf_nn_max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.max_pool2d_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.max_pool2d_7'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.max_pool2d', generated_inputs['tf.nn.max_pool2d_7'], lib="tf", suffix=7)
