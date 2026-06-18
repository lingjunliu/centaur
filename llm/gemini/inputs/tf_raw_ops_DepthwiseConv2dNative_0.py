
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DepthwiseConv2dNative_inputs():
    list_of_inputs = []

    # Input 1
    input_val = np.random.uniform(-1.0, 1.0, (1, 5, 5, 2)).astype(np.float32)
    filter_val = np.random.uniform(-1.0, 1.0, (3, 3, 2, 2)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter': filter_val,
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_val = np.random.uniform(-1.0, 1.0, (2, 10, 10, 3)).astype(np.float64)
    filter_val = np.random.uniform(-1.0, 1.0, (5, 5, 3, 1)).astype(np.float64)
    input_dict = {
        'input': input_val,
        'filter': filter_val,
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_val = np.random.uniform(-1.0, 1.0, (1, 8, 8, 1)).astype(np.float32)
    filter_val = np.random.uniform(-1.0, 1.0, (2, 2, 1, 3)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter': filter_val,
        'strides': [1, 1, 1, 1],
        'padding': 'EXPLICIT',
        'explicit_paddings': [0, 0, 1, 1, 1, 1, 0, 0],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_val = np.random.uniform(-1.0, 1.0, (2, 8, 8, 3)).astype(np.float32)
    filter_val = np.random.uniform(-1.0, 1.0, (3, 3, 3, 2)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter': filter_val,
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_val = np.random.uniform(-1.0, 1.0, (1, 4, 4, 4)).astype(np.float16)
    filter_val = np.random.uniform(-1.0, 1.0, (2, 2, 4, 1)).astype(np.float16)
    input_dict = {
        'input': input_val,
        'filter': filter_val,
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_val = np.random.uniform(-1.0, 1.0, (4, 6, 6, 2)).astype(np.float32)
    filter_val = np.random.uniform(-1.0, 1.0, (3, 3, 2, 3)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter': filter_val,
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_val = np.random.uniform(-1.0, 1.0, (1, 12, 12, 3)).astype(np.float32)
    filter_val = np.random.uniform(-1.0, 1.0, (5, 5, 3, 2)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter': filter_val,
        'strides': [1, 3, 3, 1],
        'padding': 'SAME',
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_val = np.random.uniform(-1.0, 1.0, (2, 8, 8, 2)).astype(np.float64)
    filter_val = np.random.uniform(-1.0, 1.0, (3, 3, 2, 2)).astype(np.float64)
    input_dict = {
        'input': input_val,
        'filter': filter_val,
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_val = np.random.uniform(-1.0, 1.0, (1, 3, 3, 1)).astype(np.float32)
    filter_val = np.random.uniform(-1.0, 1.0, (1, 1, 1, 1)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter': filter_val,
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_val = np.random.uniform(-1.0, 1.0, (2, 14, 14, 2)).astype(np.float32)
    filter_val = np.random.uniform(-1.0, 1.0, (3, 3, 2, 4)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter': filter_val,
        'strides': [1, 2, 2, 1],
        'padding': 'EXPLICIT',
        'explicit_paddings': [0, 0, 2, 2, 2, 2, 0, 0],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.DepthwiseConv2dNative"] = tf_raw_ops_DepthwiseConv2dNative_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DepthwiseConv2dNative' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DepthwiseConv2dNative'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.DepthwiseConv2dNative', generated_inputs['tf.raw_ops.DepthwiseConv2dNative'], lib="tf", suffix=0)
