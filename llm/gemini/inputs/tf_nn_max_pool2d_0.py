
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
        'input': np.random.randn(1, 4, 4, 1).astype(np.float32),
        'ksize': [2, 2],
        'strides': [2, 2],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'name': 'pool1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'input': np.random.randn(2, 8, 8, 3).astype(np.float32),
        'ksize': [3, 3],
        'strides': [1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'name': 'pool2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'input': np.random.randn(1, 3, 16, 16).astype(np.float32),
        'ksize': [2, 2],
        'strides': [2, 2],
        'padding': 'VALID',
        'data_format': 'NCHW',
        'name': 'pool3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'input': np.random.randn(4, 2, 5, 5).astype(np.float32),
        'ksize': [2, 2],
        'strides': [1, 1],
        'padding': 'SAME',
        'data_format': 'NCHW',
        'name': 'pool4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'input': np.random.uniform(-10.0, 10.0, size=(1, 6, 6, 2)).astype(np.float32),
        'ksize': [2, 2],
        'strides': [2, 2],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'name': 'pool5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'input': np.random.randn(2, 10, 10, 4).astype(np.float32),
        'ksize': [4, 4],
        'strides': [2, 2],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'name': 'pool6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'input': np.random.randn(1, 1, 28, 28).astype(np.float32),
        'ksize': [3, 3],
        'strides': [2, 2],
        'padding': 'SAME',
        'data_format': 'NCHW',
        'name': 'pool7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'input': np.random.randn(3, 14, 14, 8).astype(np.float32),
        'ksize': [2, 2],
        'strides': [2, 2],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'name': 'pool8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'input': np.random.randn(1, 1, 1, 1).astype(np.float32),
        'ksize': [1, 1],
        'strides': [1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'name': 'pool9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'input': np.random.randn(2, 4, 32, 32).astype(np.float32),
        'ksize': [2, 2],
        'strides': [1, 1],
        'padding': 'SAME',
        'data_format': 'NCHW',
        'name': 'pool10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.max_pool2d"] = tf_nn_max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.max_pool2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.max_pool2d'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.max_pool2d', generated_inputs['tf.nn.max_pool2d'], lib="tf", suffix=0)
