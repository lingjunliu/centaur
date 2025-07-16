
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_conv_transpose_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 3, 3).astype(np.float32)
    output_shape_tensor = np.array([1, 7, 7, 3], dtype=np.int32)
    strides_list = [1, 1, 1, 1]
    padding_string = 'SAME'
    data_format_string = 'NHWC'
    dilations_list = [1, 1, 1, 1]
    name_string = 'transpose_conv1'

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "output_shape": output_shape_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(2, 10, 10, 1).astype(np.float32)
    filters_tensor = np.random.rand(5, 5, 1, 1).astype(np.float32)
    output_shape_tensor = np.array([2, 14, 14, 1], dtype=np.int32)
    strides_list = [1, 2, 2, 1]
    padding_string = 'VALID'
    data_format_string = 'NHWC'
    dilations_list = [1, 1, 1, 1]
    name_string = 'transpose_conv2'

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "output_shape": output_shape_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(1, 8, 8, 64).astype(np.float32)
    filters_tensor = np.random.rand(2, 2, 64, 64).astype(np.float32)
    output_shape_tensor = np.array([1, 16, 16, 64], dtype=np.int32)
    strides_list = [1, 2, 2, 1]
    padding_string = 'SAME'
    data_format_string = 'NHWC'
    dilations_list = [1, 1, 1, 1]
    name_string = 'transpose_conv3'

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "output_shape": output_shape_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(4, 4, 4, 16).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 16, 16).astype(np.float32)
    output_shape_tensor = np.array([4, 6, 6, 16], dtype=np.int32)
    strides_list = [1, 1, 1, 1]
    padding_string = 'VALID'
    data_format_string = 'NHWC'
    dilations_list = [1, 1, 1, 1] #Reverted dilation back to one
    name_string = 'transpose_conv4'

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "output_shape": output_shape_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(1, 16, 16, 3).astype(np.float32)
    filters_tensor = np.random.rand(4, 4, 3, 3).astype(np.float32)
    output_shape_tensor = np.array([1, 32, 32, 3], dtype=np.int32)
    strides_list = [1, 2, 2, 1]
    padding_string = 'SAME'
    data_format_string = 'NHWC'
    dilations_list = [1, 1, 1, 1]
    name_string = 'transpose_conv5'

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "output_shape": output_shape_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.conv_transpose"] = tf_nn_conv_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.conv_transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.conv_transpose'.")

check_valid('tf.nn.conv_transpose', generated_inputs['tf.nn.conv_transpose'], lib="tf", suffix=0)
