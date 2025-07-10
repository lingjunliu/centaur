
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
    output_shape_tensor = np.array([1, 7, 7, 3]).astype(np.int32)
    strides_list = [1, 1, 1, 1]
    padding_string = 'SAME'
    data_format_string = 'NHWC'
    dilations_list = [1, 1, 1, 1]
    name_string = 'conv_transpose_1'
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
    filters_tensor = np.random.rand(5, 5, 1, 2).astype(np.float32)
    output_shape_tensor = np.array([2, 14, 14, 2]).astype(np.int32)
    strides_list = [1, 2, 2, 1]
    padding_string = 'VALID'
    data_format_string = 'NHWC'
    dilations_list = [1, 1, 1, 1]
    name_string = 'conv_transpose_2'
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
    input_tensor = np.random.rand(1, 8, 8, 3).astype(np.float32)
    filters_tensor = np.random.rand(2, 2, 3, 4).astype(np.float32)
    output_shape_tensor = np.array([1, 9, 9, 4]).astype(np.int32)
    strides_list = [1, 1, 1, 1]
    padding_string = 'VALID'
    data_format_string = 'NHWC'
    dilations_list = [1, 2, 2, 1]
    name_string = 'conv_transpose_3'
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
    input_tensor = np.random.rand(4, 4, 4, 2).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 2, 3).astype(np.float32)
    output_shape_tensor = np.array([4, 6, 6, 3]).astype(np.int32)
    strides_list = [1, 2, 2, 1]
    padding_string = 'SAME'
    data_format_string = 'NHWC'
    dilations_list = [1, 1, 1, 1]
    name_string = 'conv_transpose_4'
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

   # Input 5: NCHW format
    input_tensor = np.random.rand(1, 3, 5, 5).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 3, 4).astype(np.float32)
    output_shape_tensor = np.array([1, 4, 7, 7]).astype(np.int32)
    strides_list = [1, 1, 1, 1]
    padding_string = 'SAME'
    data_format_string = 'NCHW'
    dilations_list = [1, 1, 1, 1]
    name_string = 'conv_transpose_5'
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

    # Input 6: NCHW format with different strides
    input_tensor = np.random.rand(2, 1, 8, 8).astype(np.float32)
    filters_tensor = np.random.rand(4, 4, 1, 2).astype(np.float32)
    output_shape_tensor = np.array([2, 2, 14, 14]).astype(np.int32)
    strides_list = [1, 1, 2, 2]
    padding_string = 'VALID'
    data_format_string = 'NCHW'
    dilations_list = [1, 1, 1, 1]
    name_string = 'conv_transpose_6'
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

    # Input 7
    input_tensor = np.random.rand(1, 3, 3, 1).astype(np.float32)
    filters_tensor = np.random.rand(2, 2, 1, 1).astype(np.float32)
    output_shape_tensor = np.array([1, 4, 4, 1]).astype(np.int32)
    strides_list = [1, 1, 1, 1]
    padding_string = 'VALID'
    data_format_string = 'NHWC'
    dilations_list = [1, 1, 1, 1]
    name_string = 'conv_transpose_7'
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

    # Input 8
    input_tensor = np.random.rand(1, 7, 7, 1).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 1, 1).astype(np.float32)
    output_shape_tensor = np.array([1, 9, 9, 1]).astype(np.int32)
    strides_list = [1, 1, 1, 1]
    padding_string = 'VALID'
    data_format_string = 'NHWC'
    dilations_list = [1, 1, 1, 1]
    name_string = 'conv_transpose_8'
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

    # Input 9
    input_tensor = np.random.rand(1, 5, 5, 1).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 1, 1).astype(np.float32)
    output_shape_tensor = np.array([1, 5, 5, 1]).astype(np.int32)
    strides_list = [1, 1, 1, 1]
    padding_string = 'SAME'
    data_format_string = 'NHWC'
    dilations_list = [1, 1, 1, 1]
    name_string = 'conv_transpose_9'
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

    # Input 10
    input_tensor = np.random.rand(2, 6, 6, 2).astype(np.float32)
    filters_tensor = np.random.rand(4, 4, 2, 2).astype(np.float32)
    output_shape_tensor = np.array([2, 8, 8, 2]).astype(np.int32)
    strides_list = [1, 1, 1, 1]
    padding_string = 'VALID'
    data_format_string = 'NHWC'
    dilations_list = [1, 1, 1, 1]
    name_string = 'conv_transpose_10'
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
