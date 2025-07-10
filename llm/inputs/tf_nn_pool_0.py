
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_pool_inputs():
    list_of_inputs = []

    def to_numpy(tensor):
        return tensor.numpy()

    # Input 1: Basic 2D pooling
    input1 = np.random.rand(1, 32, 32, 3).astype(np.float32)
    input_dict = {
        'input': tf.convert_to_tensor(input1),
        'window_shape': [2, 2],
        'pooling_type': 'MAX',
        'strides': [2, 2],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'max_pool_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic 2D average pooling with SAME padding
    input2 = np.random.rand(1, 16, 16, 3).astype(np.float32)
    input_dict = {
        'input': tf.convert_to_tensor(input2),
        'window_shape': [3, 3],
        'pooling_type': 'AVG',
        'strides': [1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'avg_pool_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D pooling
    input3 = np.random.rand(1, 64, 1).astype(np.float32)
    input_dict = {
        'input': tf.convert_to_tensor(input3),
        'window_shape': [4],
        'pooling_type': 'MAX',
        'strides': [2],
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'max_pool_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D pooling
    input4 = np.random.rand(1, 8, 8, 8, 3).astype(np.float32)
    input_dict = {
        'input': tf.convert_to_tensor(input4),
        'window_shape': [2, 2, 2],
        'pooling_type': 'AVG',
        'strides': [2, 2, 2],
        'padding': 'VALID',
        'data_format': 'NDHWC',
        'dilations': [1, 1, 1],
        'name': 'avg_pool_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5:  NCHW data format
    input5 = np.random.rand(1, 3, 32, 32).astype(np.float32)
    input_dict = {
        'input': tf.convert_to_tensor(input5),
        'window_shape': [2, 2],
        'pooling_type': 'MAX',
        'strides': [2, 2],
        'padding': 'VALID',
        'data_format': 'NCHW',
        'dilations': [1, 1],
        'name': 'max_pool_nchw'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6:  NCDHW data format
    input6 = np.random.rand(1, 3, 8, 8, 8).astype(np.float32)
    input_dict = {
        'input': tf.convert_to_tensor(input6),
        'window_shape': [2, 2, 2],
        'pooling_type': 'AVG',
        'strides': [2, 2, 2],
        'padding': 'VALID',
        'data_format': 'NCDHW',
        'dilations': [1, 1, 1],
        'name': 'avg_pool_ncdhw'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7:  Strides = 1
    input7 = np.random.rand(1, 32, 32, 3).astype(np.float32)
    input_dict = {
        'input': tf.convert_to_tensor(input7),
        'window_shape': [3, 3],
        'pooling_type': 'MAX',
        'strides': [1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'max_pool_strides_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.pool"] = tf_nn_pool_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.pool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.pool'.")

check_valid('tf.nn.pool', generated_inputs['tf.nn.pool'], lib="tf", suffix=0)
