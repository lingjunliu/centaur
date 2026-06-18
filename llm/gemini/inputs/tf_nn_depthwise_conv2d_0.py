
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_depthwise_conv2d_inputs():
    list_of_inputs = []

    # Input 1: Standard case NHWC, stride 1, no dilation, valid padding, multiplier 1
    list_of_inputs.append({
        'input': np.random.randn(2, 5, 5, 3).astype(np.float32),
        'filter': np.random.randn(3, 3, 3, 1).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'depthwise_conv_1'
    })

    # Input 2: NHWC, stride 2, valid padding, multiplier 2
    list_of_inputs.append({
        'input': np.random.randn(1, 10, 10, 2).astype(np.float32),
        'filter': np.random.randn(3, 3, 2, 2).astype(np.float32),
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'depthwise_conv_2'
    })

    # Input 3: NHWC, SAME padding, stride 1, multiplier 1
    list_of_inputs.append({
        'input': np.random.randn(4, 8, 8, 4).astype(np.float32),
        'filter': np.random.randn(5, 5, 4, 1).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'depthwise_conv_3'
    })

    # Input 4: NCHW format, SAME padding, stride 1
    list_of_inputs.append({
        'input': np.random.randn(2, 3, 16, 16).astype(np.float32),
        'filter': np.random.randn(3, 3, 3, 2).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NCHW',
        'dilations': [1, 1],
        'name': 'depthwise_conv_4'
    })

    # Input 5: Dilated NHWC, stride 1, SAME padding, dilation [2, 2]
    list_of_inputs.append({
        'input': np.random.randn(1, 14, 14, 3).astype(np.float32),
        'filter': np.random.randn(3, 3, 3, 1).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [2, 2],
        'name': 'depthwise_conv_5'
    })

    # Input 6: Negative values, NHWC, VALID, stride 1, multiplier 3
    list_of_inputs.append({
        'input': -np.random.rand(2, 4, 4, 2).astype(np.float32),
        'filter': -np.random.rand(2, 2, 2, 3).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'depthwise_conv_6'
    })

    # Input 7: Float64 type
    list_of_inputs.append({
        'input': np.random.randn(1, 5, 5, 1).astype(np.float64),
        'filter': np.random.randn(3, 3, 1, 2).astype(np.float64),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'depthwise_conv_7'
    })

    # Input 8: High dilation, NCHW
    list_of_inputs.append({
        'input': np.random.randn(1, 2, 32, 32).astype(np.float32),
        'filter': np.random.randn(5, 5, 2, 1).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NCHW',
        'dilations': [3, 3],
        'name': 'depthwise_conv_8'
    })

    # Input 9: Stride 3, SAME padding
    list_of_inputs.append({
        'input': np.random.randn(2, 15, 15, 3).astype(np.float32),
        'filter': np.random.randn(3, 3, 3, 2).astype(np.float32),
        'strides': [1, 3, 3, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'depthwise_conv_9'
    })

    # Input 10: Multiplier 5, small height/width
    list_of_inputs.append({
        'input': np.random.randn(3, 3, 3, 4).astype(np.float32),
        'filter': np.random.randn(1, 1, 4, 5).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'depthwise_conv_10'
    })

    return list_of_inputs

generated_inputs["tf.nn.depthwise_conv2d"] = tf_nn_depthwise_conv2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.depthwise_conv2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.depthwise_conv2d'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.depthwise_conv2d', generated_inputs['tf.nn.depthwise_conv2d'], lib="tf", suffix=0)
