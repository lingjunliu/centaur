
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool3d_inputs():
    list_of_inputs = []

    # Input 1, valid, 5-D tensor, NDHWC format, float32
    input_dict = {
        "input": np.random.randn(1, 3, 3, 3, 1).astype(np.float32),
        "ksize": [1, 2, 2, 2, 1],
        "strides": [1, 1, 1, 1, 1],
        "padding": "VALID",
        "data_format": "NDHWC",
        "name": "pool_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid, SAME padding, NDHWC format, float32
    input_dict = {
        "input": np.random.randn(2, 4, 4, 4, 3).astype(np.float32),
        "ksize": [1, 2, 2, 2, 1],
        "strides": [1, 2, 2, 2, 1],
        "padding": "SAME",
        "data_format": "NDHWC",
        "name": "pool_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid, NCDHW format, VALID padding
    input_dict = {
        "input": np.random.randn(1, 2, 5, 5, 5).astype(np.float32),
        "ksize": [1, 1, 3, 3, 3],
        "strides": [1, 1, 2, 2, 2],
        "padding": "VALID",
        "data_format": "NCDHW",
        "name": "pool_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid, float32, 3-element ksize and strides list
    input_dict = {
        "input": np.random.randn(2, 2, 3, 3, 2).astype(np.float32),
        "ksize": [2, 2, 2],
        "strides": [1, 1, 1],
        "padding": "SAME",
        "data_format": "NDHWC",
        "name": "pool_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid, with larger dimension size, NDHWC
    input_dict = {
        "input": np.random.randn(1, 8, 8, 8, 1).astype(np.float32),
        "ksize": [1, 3, 3, 3, 1],
        "strides": [1, 2, 2, 2, 1],
        "padding": "SAME",
        "data_format": "NDHWC",
        "name": "pool_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid, negative values, 1-element ksize and strides list
    input_dict = {
        "input": -np.random.rand(1, 4, 4, 4, 4).astype(np.float32),
        "ksize": [1],
        "strides": [1],
        "padding": "VALID",
        "data_format": "NDHWC",
        "name": "pool_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid, NCDHW format, SAME padding, float32
    input_dict = {
        "input": np.random.randn(1, 3, 10, 10, 10).astype(np.float32),
        "ksize": [3, 3, 3],
        "strides": [2, 2, 2],
        "padding": "SAME",
        "data_format": "NCDHW",
        "name": "pool_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid, strides larger than window size
    input_dict = {
        "input": np.random.randn(3, 2, 2, 2, 1).astype(np.float32),
        "ksize": [2, 2, 2],
        "strides": [3, 3, 3],
        "padding": "VALID",
        "data_format": "NDHWC",
        "name": "pool_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid, minimum spatial dimension 1
    input_dict = {
        "input": np.random.randn(1, 1, 1, 1, 1).astype(np.float32),
        "ksize": [1, 1, 1, 1, 1],
        "strides": [1, 1, 1, 1, 1],
        "padding": "SAME",
        "data_format": "NDHWC",
        "name": "pool_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid, float32, NCDHW format with mismatched ksize and strides sizes (5 vs 3, but valid)
    input_dict = {
        "input": np.random.randn(2, 5, 2, 2, 2).astype(np.float32),
        "ksize": [1, 1, 2, 2, 2],
        "strides": [1, 1, 1, 1, 1],
        "padding": "VALID",
        "data_format": "NCDHW",
        "name": "pool_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.max_pool3d"] = tf_nn_max_pool3d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.max_pool3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.max_pool3d'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.max_pool3d', generated_inputs['tf.nn.max_pool3d'], lib="tf", suffix=0)
