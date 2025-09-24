
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_depthwise_conv2d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.float32)
    filter_tensor = np.array([[[[1, 2]]]], dtype=np.float32)
    strides_val = [1, 1, 1, 1]
    padding_val = 'VALID'
    data_format_val = 'NHWC'
    dilations_val = [1, 1]
    name_val = 'depthwise_conv2d_1'
    input_dict = {'input': input_tensor, 'filter': filter_tensor, 'strides': strides_val, 'padding': padding_val, 'data_format': data_format_val, 'dilations': dilations_val, 'name': name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.float32)
    filter_tensor = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
    strides_val = [1, 1, 1, 1]
    padding_val = 'SAME'
    data_format_val = 'NHWC'
    dilations_val = [1, 1]
    name_val = 'depthwise_conv2d_2'
    input_dict = {'input': input_tensor, 'filter': filter_tensor, 'strides': strides_val, 'padding': padding_val, 'data_format': data_format_val, 'dilations': dilations_val, 'name': name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]], dtype=np.float32)
    filter_tensor = np.array([[[[1, 2]]]], dtype=np.float32)
    strides_val = [1, 1, 1, 1]
    padding_val = 'VALID'
    data_format_val = 'NHWC'
    dilations_val = [1, 1]
    name_val = 'depthwise_conv2d_3'
    input_dict = {'input': input_tensor, 'filter': filter_tensor, 'strides': strides_val, 'padding': padding_val, 'data_format': data_format_val, 'dilations': dilations_val, 'name': name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.float32)
    filter_tensor = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
    strides_val = [1, 1, 1, 1]
    padding_val = [[0, 0], [1, 1], [1, 1], [0, 0]]
    data_format_val = 'NHWC'
    dilations_val = [1, 1]
    name_val = 'depthwise_conv2d_4'
    input_dict = {'input': input_tensor, 'filter': filter_tensor, 'strides': strides_val, 'padding': padding_val, 'data_format': data_format_val, 'dilations': dilations_val, 'name': name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[[1, 2, 3]]]], dtype=np.float32)
    filter_tensor = np.array([[[[1, 2], [3, 4], [5, 6]]]], dtype=np.float32)
    strides_val = [1, 1, 1, 1]
    padding_val = 'VALID'
    data_format_val = 'NHWC'
    dilations_val = [1, 1]
    name_val = 'depthwise_conv2d_5'
    input_dict = {'input': input_tensor, 'filter': filter_tensor, 'strides': strides_val, 'padding': padding_val, 'data_format': data_format_val, 'dilations': dilations_val, 'name': name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.float32)
    filter_tensor = np.array([[[[1, 2]]]], dtype=np.float32)
    strides_val = [1, 1, 1, 1]
    padding_val = 'SAME'
    data_format_val = 'NHWC'
    dilations_val = [1, 1]
    name_val = 'depthwise_conv2d_6'
    input_dict = {'input': input_tensor, 'filter': filter_tensor, 'strides': strides_val, 'padding': padding_val, 'data_format': data_format_val, 'dilations': dilations_val, 'name': name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[[[1, -2], [-3, 4]], [[-5, 6], [7, -8]]]], dtype=np.float32)
    filter_tensor = np.array([[[[1, 2], [3, -4]]]], dtype=np.float32)
    strides_val = [1, 1, 1, 1]
    padding_val = 'SAME'
    data_format_val = 'NHWC'
    dilations_val = [1, 1]
    name_val = 'depthwise_conv2d_7'
    input_dict = {'input': input_tensor, 'filter': filter_tensor, 'strides': strides_val, 'padding': padding_val, 'data_format': data_format_val, 'dilations': dilations_val, 'name': name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: NCHW format
    input_tensor = np.array([[[[1., 2.], [3., 4.]]]], dtype=np.float32)
    filter_tensor = np.array([[[[1., 2.]]]], dtype=np.float32)
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    data_format_val = "NCHW"
    dilations_val = [1, 1]
    name_val = "depthwise_conv2d_8"
    input_dict = {'input': input_tensor, 'filter': filter_tensor, 'strides': strides_val, 'padding': padding_val, 'data_format': data_format_val, 'dilations': dilations_val, 'name': name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.depthwise_conv2d"] = tf_nn_depthwise_conv2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.depthwise_conv2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.depthwise_conv2d'.")

check_valid('tf.nn.depthwise_conv2d', generated_inputs['tf.nn.depthwise_conv2d'], lib="tf", suffix=0)
