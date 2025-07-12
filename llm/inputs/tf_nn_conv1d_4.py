
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_conv1d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 5, 3).astype(np.float32)
    filters_tensor = np.random.rand(2, 3, 4).astype(np.float32)
    stride_val = [1]
    padding_val = 'VALID'
    data_format_val = 'NWC'
    dilations_val = 1
    name_val = 'conv1d_1'

    input_dict = {
        'input': input_tensor,
        'filters': filters_tensor,
        'stride': stride_val,
        'padding': padding_val,
        'data_format': data_format_val,
        'dilations': dilations_val,
        'name': name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(2, 7, 5).astype(np.float32)
    filters_tensor = np.random.rand(3, 5, 2).astype(np.float32)
    stride_val = [2]
    padding_val = 'SAME'
    data_format_val = 'NWC'
    dilations_val = 1
    name_val = 'conv1d_2'

    input_dict = {
        'input': input_tensor,
        'filters': filters_tensor,
        'stride': stride_val,
        'padding': padding_val,
        'data_format': data_format_val,
        'dilations': dilations_val,
        'name': name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(1, 10, 1).astype(np.float32)
    filters_tensor = np.random.rand(4, 1, 8).astype(np.float32)
    stride_val = [3]
    padding_val = 'VALID'
    data_format_val = 'NWC'
    dilations_val = 2
    name_val = 'conv1d_3'

    input_dict = {
        'input': input_tensor,
        'filters': filters_tensor,
        'stride': stride_val,
        'padding': padding_val,
        'data_format': data_format_val,
        'dilations': dilations_val,
        'name': name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(3, 12, 4).astype(np.float32)
    filters_tensor = np.random.rand(5, 4, 6).astype(np.float32)
    stride_val = [1]
    padding_val = 'SAME'
    data_format_val = 'NWC'
    dilations_val = 3
    name_val = 'conv1d_4'

    input_dict = {
        'input': input_tensor,
        'filters': filters_tensor,
        'stride': stride_val,
        'padding': padding_val,
        'data_format': data_format_val,
        'dilations': dilations_val,
        'name': name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(1, 8, 2).astype(np.float32)
    filters_tensor = np.random.rand(2, 2, 3).astype(np.float32)
    stride_val = [4]
    padding_val = 'VALID'
    data_format_val = 'NWC'
    dilations_val = 1
    name_val = 'conv1d_5'

    input_dict = {
        'input': input_tensor,
        'filters': filters_tensor,
        'stride': stride_val,
        'padding': padding_val,
        'data_format': data_format_val,
        'dilations': dilations_val,
        'name': name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.rand(2, 6, 7).astype(np.float32)
    filters_tensor = np.random.rand(3, 7, 1).astype(np.float32)
    stride_val = [2]
    padding_val = 'SAME'
    data_format_val = 'NWC'
    dilations_val = 2
    name_val = 'conv1d_6'

    input_dict = {
        'input': input_tensor,
        'filters': filters_tensor,
        'stride': stride_val,
        'padding': padding_val,
        'data_format': data_format_val,
        'dilations': dilations_val,
        'name': name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: NCW data format
    input_tensor = np.random.rand(1, 3, 5).astype(np.float32)
    filters_tensor = np.random.rand(2, 3, 4).astype(np.float32)
    stride_val = [1]
    padding_val = 'VALID'
    data_format_val = 'NCW'
    dilations_val = 1
    name_val = 'conv1d_7'

    input_dict = {
        'input': input_tensor,
        'filters': filters_tensor,
        'stride': stride_val,
        'padding': padding_val,
        'data_format': data_format_val,
        'dilations': dilations_val,
        'name': name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: NCW data format, different stride
    input_tensor = np.random.rand(2, 5, 7).astype(np.float32)
    filters_tensor = np.random.rand(3, 5, 2).astype(np.float32)
    stride_val = [2]
    padding_val = 'SAME'
    data_format_val = 'NCW'
    dilations_val = 1
    name_val = 'conv1d_8'

    input_dict = {
        'input': input_tensor,
        'filters': filters_tensor,
        'stride': stride_val,
        'padding': padding_val,
        'data_format': data_format_val,
        'dilations': dilations_val,
        'name': name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9: Float16
    input_tensor = np.random.rand(1, 5, 3).astype(np.float16)
    filters_tensor = np.random.rand(2, 3, 4).astype(np.float16)
    stride_val = [1]
    padding_val = 'VALID'
    data_format_val = 'NWC'
    dilations_val = 1
    name_val = 'conv1d_9'

    input_dict = {
        'input': input_tensor,
        'filters': filters_tensor,
        'stride': stride_val,
        'padding': padding_val,
        'data_format': data_format_val,
        'dilations': dilations_val,
        'name': name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float64
    input_tensor = np.random.rand(2, 7, 5).astype(np.float64)
    filters_tensor = np.random.rand(3, 5, 2).astype(np.float64)
    stride_val = [2]
    padding_val = 'SAME'
    data_format_val = 'NWC'
    dilations_val = 1
    name_val = 'conv1d_10'

    input_dict = {
        'input': input_tensor,
        'filters': filters_tensor,
        'stride': stride_val,
        'padding': padding_val,
        'data_format': data_format_val,
        'dilations': dilations_val,
        'name': name_val
    }
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
