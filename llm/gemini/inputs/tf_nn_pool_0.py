
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_pool_inputs():
    list_of_inputs = []

    # Input 1: N=1, NWC, MAX, VALID
    input_dict = {
        'input': np.random.randn(2, 10, 3).astype(np.float32),
        'window_shape': [3],
        'pooling_type': 'MAX',
        'strides': [1],
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'pool_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: N=1, NCW, AVG, SAME
    input_dict = {
        'input': np.random.randn(2, 3, 10).astype(np.float32),
        'window_shape': [2],
        'pooling_type': 'AVG',
        'strides': [2],
        'padding': 'SAME',
        'data_format': 'NCW',
        'dilations': [1],
        'name': 'pool_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: N=2, NHWC, MAX, SAME
    input_dict = {
        'input': np.random.randn(2, 8, 8, 3).astype(np.float32),
        'window_shape': [2, 2],
        'pooling_type': 'MAX',
        'strides': [2, 2],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'pool_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: N=2, NCHW, AVG, VALID, dilation > 1
    input_dict = {
        'input': np.random.randn(2, 3, 8, 8).astype(np.float32),
        'window_shape': [3, 3],
        'pooling_type': 'AVG',
        'strides': [1, 1],
        'padding': 'VALID',
        'data_format': 'NCHW',
        'dilations': [2, 2],
        'name': 'pool_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: N=3, NDHWC, MAX, SAME
    input_dict = {
        'input': np.random.randn(2, 4, 4, 4, 3).astype(np.float32),
        'window_shape': [2, 2, 2],
        'pooling_type': 'MAX',
        'strides': [1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NDHWC',
        'dilations': [1, 1, 1],
        'name': 'pool_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: N=3, NCDHW, AVG, VALID
    input_dict = {
        'input': np.random.randn(2, 3, 4, 4, 4).astype(np.float32),
        'window_shape': [2, 2, 2],
        'pooling_type': 'AVG',
        'strides': [2, 2, 2],
        'padding': 'VALID',
        'data_format': 'NCDHW',
        'dilations': [1, 1, 1],
        'name': 'pool_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: N=2, NHWC, asymmetric strides
    input_dict = {
        'input': np.random.randn(1, 10, 12, 4).astype(np.float32),
        'window_shape': [3, 2],
        'pooling_type': 'MAX',
        'strides': [2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'pool_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: N=2, NCHW, dilation > 1, VALID (SAME not supported for dilation > 1)
    input_dict = {
        'input': np.random.randn(1, 2, 10, 10).astype(np.float32),
        'window_shape': [3, 3],
        'pooling_type': 'MAX',
        'strides': [1, 1],
        'padding': 'VALID',
        'data_format': 'NCHW',
        'dilations': [2, 2],
        'name': 'pool_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: N=1, NWC, MAX, negative values
    input_dict = {
        'input': np.random.uniform(-10.0, -1.0, (4, 15, 2)).astype(np.float32),
        'window_shape': [4],
        'pooling_type': 'MAX',
        'strides': [3],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'pool_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: N=2, NHWC, AVG, VALID, large input
    input_dict = {
        'input': np.random.randn(1, 16, 16, 1).astype(np.float32),
        'window_shape': [4, 4],
        'pooling_type': 'AVG',
        'strides': [1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [3, 3],
        'name': 'pool_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.pool"] = tf_nn_pool_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.pool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.pool'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.pool', generated_inputs['tf.nn.pool'], lib="tf", suffix=0)
