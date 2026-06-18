
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_nn_depthwise_conv2d_backprop_filter_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'input': np.random.randn(2, 5, 5, 3).astype(np.float32),
        'filter_sizes': np.array([3, 3, 3, 2], dtype=np.int32),
        'out_backprop': np.random.randn(2, 3, 3, 6).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'depthwise_conv_backprop_filter_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'input': np.random.randn(1, 6, 6, 2).astype(np.float32),
        'filter_sizes': np.array([2, 2, 2, 1], dtype=np.int32),
        'out_backprop': np.random.randn(1, 3, 3, 2).astype(np.float32),
        'strides': [1, 2, 2, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'depthwise_conv_backprop_filter_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'input': np.random.randn(3, 4, 4, 2).astype(np.float32),
        'filter_sizes': np.array([2, 2, 2, 3], dtype=np.int32),
        'out_backprop': np.random.randn(3, 3, 3, 6).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'depthwise_conv_backprop_filter_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'input': np.random.randn(2, 8, 8, 2).astype(np.float32),
        'filter_sizes': np.array([3, 3, 2, 1], dtype=np.int32),
        'out_backprop': np.random.randn(2, 8, 8, 2).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'depthwise_conv_backprop_filter_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'input': np.random.randn(1, 3, 3, 1).astype(np.float32),
        'filter_sizes': np.array([2, 2, 1, 1], dtype=np.int32),
        'out_backprop': np.random.randn(1, 2, 2, 1).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'depthwise_conv_backprop_filter_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'input': np.random.randn(1, 7, 7, 1).astype(np.float32),
        'filter_sizes': np.array([3, 3, 1, 2], dtype=np.int32),
        'out_backprop': np.random.randn(1, 4, 4, 2).astype(np.float32),
        'strides': [1, 2, 2, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'depthwise_conv_backprop_filter_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'input': np.random.randn(2, 5, 5, 2).astype(np.float32),
        'filter_sizes': np.array([3, 3, 2, 2], dtype=np.int32),
        'out_backprop': np.random.randn(2, 2, 2, 4).astype(np.float32),
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'depthwise_conv_backprop_filter_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'input': np.random.randn(1, 10, 10, 2).astype(np.float32),
        'filter_sizes': np.array([3, 3, 2, 1], dtype=np.int32),
        'out_backprop': np.random.randn(1, 4, 4, 2).astype(np.float32),
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'depthwise_conv_backprop_filter_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'input': -np.random.rand(1, 4, 4, 1).astype(np.float32),
        'filter_sizes': np.array([2, 2, 1, 1], dtype=np.int32),
        'out_backprop': -np.random.rand(1, 4, 4, 1).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'depthwise_conv_backprop_filter_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'input': np.random.randn(1, 10, 10, 3).astype(np.float32),
        'filter_sizes': np.array([4, 4, 3, 1], dtype=np.int32),
        'out_backprop': np.random.randn(1, 3, 3, 3).astype(np.float32),
        'strides': [1, 3, 3, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'depthwise_conv_backprop_filter_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.depthwise_conv2d_backprop_filter"] = tf_nn_depthwise_conv2d_backprop_filter_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.depthwise_conv2d_backprop_filter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.depthwise_conv2d_backprop_filter'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.depthwise_conv2d_backprop_filter', generated_inputs['tf.nn.depthwise_conv2d_backprop_filter'], lib="tf", suffix=0)
