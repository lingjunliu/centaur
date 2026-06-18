
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_conv2d_inputs():
    list_of_inputs = []

    # Input 1: Standard NHWC, float32, SAME padding
    input_dict = {
        'input': np.random.randn(1, 16, 16, 3).astype(np.float32),
        'filters': np.random.randn(3, 3, 3, 16).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: NHWC, float16, VALID padding, strides of 2
    input_dict = {
        'input': np.random.randn(2, 10, 10, 1).astype(np.float16),
        'filters': np.random.randn(2, 2, 1, 8).astype(np.float16),
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: NHWC format, float64, strides length 4
    input_dict = {
        'input': np.random.randn(1, 32, 32, 4).astype(np.float64),
        'filters': np.random.randn(3, 3, 4, 8).astype(np.float64),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: NHWC, dilated convolution with dilation 2
    input_dict = {
        'input': np.random.randn(4, 16, 16, 3).astype(np.float32),
        'filters': np.random.randn(3, 3, 3, 8).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 2, 2, 1],
        'name': 'conv_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values, NHWC, float32, VALID padding
    input_dict = {
        'input': (np.ones((1, 8, 8, 1), dtype=np.float32) * -1.5),
        'filters': (np.ones((2, 2, 1, 1), dtype=np.float32) * 0.5),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Higher dimension batch (5-D input)
    input_dict = {
        'input': np.random.randn(2, 3, 10, 10, 3).astype(np.float32),
        'filters': np.random.randn(2, 2, 3, 4).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Strides of length 2
    input_dict = {
        'input': np.random.randn(1, 14, 14, 4).astype(np.float32),
        'filters': np.random.randn(3, 3, 4, 16).astype(np.float32),
        'strides': [2, 2],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'conv_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Strides of length 1 (replicated to H and W)
    input_dict = {
        'input': np.random.randn(2, 8, 8, 2).astype(np.float32),
        'filters': np.random.randn(2, 2, 2, 4).astype(np.float32),
        'strides': [1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1],
        'name': 'conv_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: High dimension batch (6-D input)
    input_dict = {
        'input': np.random.randn(1, 2, 2, 12, 12, 3).astype(np.float32),
        'filters': np.random.randn(3, 3, 3, 6).astype(np.float32),
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1x1 Convolution with many channels
    input_dict = {
        'input': np.random.randn(1, 8, 8, 64).astype(np.float32),
        'filters': np.random.randn(1, 1, 64, 128).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.conv2d"] = tf_nn_conv2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.conv2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.conv2d'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.conv2d', generated_inputs['tf.nn.conv2d'], lib="tf", suffix=0)
