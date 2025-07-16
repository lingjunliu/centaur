
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_conv1d_transpose_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 5, 3).astype(np.float32)
    filters_tensor = np.random.rand(3, 2, 3).astype(np.float32)
    output_shape_tensor = np.array([1, 7, 2]).astype(np.int32)
    strides_int = 1
    padding_str = 'SAME'
    data_format_str = 'NWC'
    dilations_int = 1
    name_str = 'conv1d_transpose_1'
    input_dict = {'input': input_tensor, 'filters': filters_tensor, 'output_shape': output_shape_tensor, 'strides': strides_int, 'padding': padding_str, 'data_format': data_format_str, 'dilations': dilations_int, 'name': name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(1, 5, 3).astype(np.float32)
    filters_tensor = np.random.rand(3, 2, 3).astype(np.float32)
    output_shape_tensor = np.array([1, 9, 2]).astype(np.int32)
    strides_int = 2
    padding_str = 'VALID'
    data_format_str = 'NWC'
    dilations_int = 1
    name_str = 'conv1d_transpose_2'
    input_dict = {'input': input_tensor, 'filters': filters_tensor, 'output_shape': output_shape_tensor, 'strides': strides_int, 'padding': padding_str, 'data_format': data_format_str, 'dilations': dilations_int, 'name': name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.conv1d_transpose"] = tf_nn_conv1d_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.conv1d_transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.conv1d_transpose'.")

check_valid('tf.nn.conv1d_transpose', generated_inputs['tf.nn.conv1d_transpose'], lib="tf", suffix=0)
