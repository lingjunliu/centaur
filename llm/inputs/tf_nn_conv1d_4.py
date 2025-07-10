
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_conv1d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 10, 3).astype(np.float32)
    filter_tensor = np.random.rand(3, 3, 3).astype(np.float32)
    stride = [1]
    padding = 'VALID'
    data_format = 'NWC'
    dilations = 1
    name = 'conv1d_1'
    input_dict = {'input': input_tensor, 'filters': filter_tensor, 'stride': stride, 'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(2, 20, 4).astype(np.float32)
    filter_tensor = np.random.rand(5, 4, 4).astype(np.float32)
    stride = [2]
    padding = 'SAME'
    data_format = 'NWC'
    dilations = 1
    name = 'conv1d_2'
    input_dict = {'input': input_tensor, 'filters': filter_tensor, 'stride': stride, 'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(1, 15, 2).astype(np.float32)
    filter_tensor = np.random.rand(4, 2, 2).astype(np.float32)
    stride = [1]
    padding = 'VALID'
    data_format = 'NWC'
    dilations = 2
    name = 'conv1d_3'
    input_dict = {'input': input_tensor, 'filters': filter_tensor, 'stride': stride, 'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(3, 12, 5).astype(np.float32)
    filter_tensor = np.random.rand(2, 5, 5).astype(np.float32)
    stride = [3]
    padding = 'SAME'
    data_format = 'NWC'
    dilations = 1
    name = 'conv1d_4'
    input_dict = {'input': input_tensor, 'filters': filter_tensor, 'stride': stride, 'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(1, 8, 3).astype(np.float32)
    filter_tensor = np.random.rand(3, 3, 3).astype(np.float32)
    stride = [1]
    padding = 'VALID'
    data_format = 'NWC'
    dilations = 3
    name = 'conv1d_5'
    input_dict = {'input': input_tensor, 'filters': filter_tensor, 'stride': stride, 'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.rand(1, 5, 3).astype(np.float32)
    filter_tensor = np.random.rand(2, 3, 4).astype(np.float32)
    stride = [1]
    padding = 'SAME'
    data_format = 'NWC'
    dilations = 1
    name = 'conv1d_6'
    input_dict = {'input': input_tensor, 'filters': filter_tensor, 'stride': stride, 'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    input_tensor = np.random.rand(1, 10, 3).astype(np.float16)
    filter_tensor = np.random.rand(3, 3, 3).astype(np.float16)
    stride = [1]
    padding = 'VALID'
    data_format = 'NWC'
    dilations = 1
    name = 'conv1d_7'
    input_dict = {'input': input_tensor, 'filters': filter_tensor, 'stride': stride, 'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.random.rand(2, 20, 4).astype(np.float64)
    filter_tensor = np.random.rand(5, 4, 4).astype(np.float64)
    stride = [2]
    padding = 'SAME'
    data_format = 'NWC'
    dilations = 1
    name = 'conv1d_8'
    input_dict = {'input': input_tensor, 'filters': filter_tensor, 'stride': stride, 'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_tensor = np.random.rand(1, 10, 3).astype(np.float32)
    filter_tensor = np.random.rand(3, 3, 3).astype(np.float32)
    stride = [1]
    padding = 'VALID'
    data_format = 'NCW'
    dilations = 1
    name = 'conv1d_9'
    input_dict = {'input': input_tensor, 'filters': filter_tensor, 'stride': stride, 'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_tensor = np.random.rand(2, 20, 4).astype(np.float32)
    filter_tensor = np.random.rand(5, 4, 4).astype(np.float32)
    stride = [2]
    padding = 'SAME'
    data_format = 'NCW'
    dilations = 1
    name = 'conv1d_10'
    input_dict = {'input': input_tensor, 'filters': filter_tensor, 'stride': stride, 'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.conv1d_4"] = tf_nn_conv1d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.conv1d_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.conv1d_4'.")

check_valid('tf.nn.conv1d', generated_inputs['tf.nn.conv1d_4'], lib="tf", suffix=4)
