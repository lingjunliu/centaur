
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MaxPool3D_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool_1',
        'input': np.random.randn(1, 2, 2, 2, 1).astype(np.float32),
        'ksize': [1, 2, 2, 2, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool_2',
        'input': np.random.randn(2, 3, 3, 3, 2).astype(np.float32),
        'ksize': [1, 2, 2, 2, 1],
        'strides': [1, 2, 2, 2, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool_3',
        'input': np.random.randn(1, 4, 4, 4, 3).astype(np.float32),
        'ksize': [1, 3, 3, 3, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool_4',
        'input': np.random.randn(1, 2, 3, 4, 1).astype(np.float32),
        'ksize': [1, 2, 2, 2, 1],
        'strides': [1, 1, 2, 2, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool_5',
        'input': np.random.randn(2, 2, 2, 2, 2).astype(np.float32),
        'ksize': [1, 1, 1, 1, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool_6',
        'input': np.random.randn(3, 5, 5, 5, 2).astype(np.float32),
        'ksize': [1, 2, 2, 2, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool_7',
        'input': np.random.randn(1, 3, 3, 3, 4).astype(np.float32),
        'ksize': [1, 2, 3, 2, 1],
        'strides': [1, 2, 1, 2, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool_8',
        'input': np.random.randn(2, 4, 4, 4, 1).astype(np.float32),
        'ksize': [1, 4, 4, 4, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool_9',
        'input': np.random.randn(1, 1, 1, 1, 1).astype(np.float32),
        'ksize': [1, 1, 1, 1, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool_10',
        'input': np.random.randn(2, 3, 4, 5, 2).astype(np.float32),
        'ksize': [1, 1, 2, 3, 1],
        'strides': [1, 1, 2, 2, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.MaxPool3D"] = tf_raw_ops_MaxPool3D_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MaxPool3D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPool3D'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.MaxPool3D', generated_inputs['tf.raw_ops.MaxPool3D'], lib="tf", suffix=0)
