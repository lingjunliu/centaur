
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_nn_conv1d_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'input': np.random.randn(2, 10, 3).astype(np.float32),
        'filters': np.random.randn(3, 3, 4).astype(np.float32),
        'stride': [1],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'conv1d_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'input': np.random.randn(2, 3, 10).astype(np.float32),
        'filters': np.random.randn(3, 3, 4).astype(np.float32),
        'stride': [1],
        'padding': 'SAME',
        'data_format': 'NCW',
        'dilations': [1],
        'name': 'conv1d_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'input': np.random.randn(1, 8, 2).astype(np.float16),
        'filters': np.random.randn(2, 2, 2).astype(np.float16),
        'stride': [2],
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'conv1d_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'input': np.random.randn(4, 20, 4).astype(np.float64),
        'filters': np.random.randn(5, 4, 8).astype(np.float64),
        'stride': [3],
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'conv1d_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'input': np.random.randn(3, 4, 12).astype(np.float32),
        'filters': np.random.randn(3, 4, 5).astype(np.float32),
        'stride': [1],
        'padding': 'SAME',
        'data_format': 'NCW',
        'dilations': [1],
        'name': 'conv1d_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'input': np.random.randn(2, 16, 3).astype(np.float32),
        'filters': np.random.randn(3, 3, 4).astype(np.float32),
        'stride': [1],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [2],
        'name': 'conv1d_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'input': np.random.randn(2, 10, 3).astype(np.float32),
        'filters': np.random.randn(3, 3, 4).astype(np.float32),
        'stride': [1, 2, 1],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [1, 1, 1],
        'name': 'conv1d_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'input': np.random.uniform(-5.0, -1.0, (2, 8, 2)).astype(np.float32),
        'filters': np.random.uniform(-2.0, 2.0, (3, 2, 3)).astype(np.float32),
        'stride': [1],
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'conv1d_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'input': np.random.randn(1, 100, 16).astype(np.float32),
        'filters': np.random.randn(7, 16, 32).astype(np.float32),
        'stride': [2],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'conv1d_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'input': np.random.randn(2, 10, 3).astype(np.float32),
        'filters': np.random.randn(3, 3, 4).astype(np.float32),
        'stride': [1],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [1, 2, 1],
        'name': 'conv1d_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.conv1d_2"] = tf_nn_conv1d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.conv1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.conv1d_2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.conv1d', generated_inputs['tf.nn.conv1d_2'], lib="tf", suffix=2)
