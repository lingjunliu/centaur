
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_separable_conv2d_inputs():
    list_of_inputs = []

    # Input 1: Basic NHWC format with same padding
    input_val = np.random.randn(2, 5, 5, 3).astype(np.float32)
    depthwise_filter = np.random.randn(3, 3, 3, 2).astype(np.float32)
    pointwise_filter = np.random.randn(1, 1, 6, 4).astype(np.float32)
    input_dict = {
        'input': input_val,
        'depthwise_filter': depthwise_filter,
        'pointwise_filter': pointwise_filter,
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'conv_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Strides greater than 1, valid padding
    input_val = np.random.randn(1, 10, 10, 2).astype(np.float32)
    depthwise_filter = np.random.randn(2, 2, 2, 1).astype(np.float32)
    pointwise_filter = np.random.randn(1, 1, 2, 3).astype(np.float32)
    input_dict = {
        'input': input_val,
        'depthwise_filter': depthwise_filter,
        'pointwise_filter': pointwise_filter,
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'conv_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Atrous depthwise convolution (dilations > 1)
    input_val = np.random.randn(1, 8, 8, 4).astype(np.float32)
    depthwise_filter = np.random.randn(3, 3, 4, 1).astype(np.float32)
    pointwise_filter = np.random.randn(1, 1, 4, 2).astype(np.float32)
    input_dict = {
        'input': input_val,
        'depthwise_filter': depthwise_filter,
        'pointwise_filter': pointwise_filter,
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [2, 2],
        'name': 'conv_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Double precision float64 type
    input_val = np.random.randn(2, 6, 6, 3).astype(np.float64)
    depthwise_filter = np.random.randn(2, 2, 3, 2).astype(np.float64)
    pointwise_filter = np.random.randn(1, 1, 6, 3).astype(np.float64)
    input_dict = {
        'input': input_val,
        'depthwise_filter': depthwise_filter,
        'pointwise_filter': pointwise_filter,
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'conv_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: NCHW format
    input_val = np.random.randn(2, 3, 5, 5).astype(np.float32)
    depthwise_filter = np.random.randn(3, 3, 3, 2).astype(np.float32)
    pointwise_filter = np.random.randn(1, 1, 6, 4).astype(np.float32)
    input_dict = {
        'input': input_val,
        'depthwise_filter': depthwise_filter,
        'pointwise_filter': pointwise_filter,
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NCHW',
        'dilations': [1, 1],
        'name': 'conv_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: NCHW format with stride 2
    input_val = np.random.randn(1, 2, 8, 8).astype(np.float32)
    depthwise_filter = np.random.randn(2, 2, 2, 1).astype(np.float32)
    pointwise_filter = np.random.randn(1, 1, 2, 2).astype(np.float32)
    input_dict = {
        'input': input_val,
        'depthwise_filter': depthwise_filter,
        'pointwise_filter': pointwise_filter,
        'strides': [1, 1, 2, 2],
        'padding': 'VALID',
        'data_format': 'NCHW',
        'dilations': [1, 1],
        'name': 'conv_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Uniformly distributed inputs with custom names
    input_val = np.random.uniform(-1, 1, (1, 7, 7, 3)).astype(np.float32)
    depthwise_filter = np.random.uniform(-1, 1, (3, 3, 3, 2)).astype(np.float32)
    pointwise_filter = np.random.uniform(-1, 1, (1, 1, 6, 5)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'depthwise_filter': depthwise_filter,
        'pointwise_filter': pointwise_filter,
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'conv_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger channel multiplier
    input_val = np.random.randn(2, 4, 4, 2).astype(np.float32)
    depthwise_filter = np.random.randn(2, 2, 2, 4).astype(np.float32)
    pointwise_filter = np.random.randn(1, 1, 8, 2).astype(np.float32)
    input_dict = {
        'input': input_val,
        'depthwise_filter': depthwise_filter,
        'pointwise_filter': pointwise_filter,
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'conv_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1x1 depthwise filter
    input_val = np.random.randn(1, 5, 5, 3).astype(np.float32)
    depthwise_filter = np.random.randn(1, 1, 3, 1).astype(np.float32)
    pointwise_filter = np.random.randn(1, 1, 3, 3).astype(np.float32)
    input_dict = {
        'input': input_val,
        'depthwise_filter': depthwise_filter,
        'pointwise_filter': pointwise_filter,
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'conv_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Dilation and NCHW format
    input_val = np.random.randn(2, 2, 6, 6).astype(np.float32)
    depthwise_filter = np.random.randn(3, 3, 2, 2).astype(np.float32)
    pointwise_filter = np.random.randn(1, 1, 4, 3).astype(np.float32)
    input_dict = {
        'input': input_val,
        'depthwise_filter': depthwise_filter,
        'pointwise_filter': pointwise_filter,
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NCHW',
        'dilations': [2, 2],
        'name': 'conv_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.separable_conv2d"] = tf_nn_separable_conv2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.separable_conv2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.separable_conv2d'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.separable_conv2d', generated_inputs['tf.nn.separable_conv2d'], lib="tf", suffix=0)
