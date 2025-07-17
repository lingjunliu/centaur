
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_conv_transpose_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 5, 5, 2).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 2, 2).astype(np.float32)
    output_shape_tensor = np.array([1, 7, 7, 2], dtype=np.int32)
    strides_int = 1
    padding_str = 'SAME'
    data_format_str = 'NHWC'
    dilations_int = 1
    name_str = 'transpose_conv1'

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "output_shape": output_shape_tensor,
        "strides": strides_int,
        "padding": padding_str,
        "data_format": data_format_str,
        "dilations": dilations_int,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(2, 10, 10, 1).astype(np.float32)
    filters_tensor = np.random.rand(5, 5, 1, 4).astype(np.float32)
    output_shape_tensor = np.array([2, 14, 14, 4], dtype=np.int32)
    strides_int = 2
    padding_str = 'VALID'
    data_format_str = 'NHWC'
    dilations_int = 1
    name_str = 'transpose_conv2'

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "output_shape": output_shape_tensor,
        "strides": strides_int,
        "padding": padding_str,
        "data_format": data_format_str,
        "dilations": dilations_int,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(1, 8, 8, 3).astype(np.float32)
    filters_tensor = np.random.rand(4, 4, 3, 2).astype(np.float32)
    output_shape_tensor = np.array([1, 11, 11, 2], dtype=np.int32)
    strides_int = 1
    padding_str = 'VALID'
    data_format_str = 'NHWC'
    dilations_int = 1
    name_str = 'transpose_conv3'

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "output_shape": output_shape_tensor,
        "strides": strides_int,
        "padding": padding_str,
        "data_format": data_format_str,
        "dilations": dilations_int,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(4, 6, 6, 2).astype(np.float32)
    filters_tensor = np.random.rand(2, 2, 2, 4).astype(np.float32)
    output_shape_tensor = np.array([4, 7, 7, 4], dtype=np.int32)
    strides_int = 1
    padding_str = 'SAME'
    data_format_str = 'NHWC'
    dilations_int = 1
    name_str = 'transpose_conv4'

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "output_shape": output_shape_tensor,
        "strides": strides_int,
        "padding": padding_str,
        "data_format": data_format_str,
        "dilations": dilations_int,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(1, 3, 3, 1).astype(np.float32)
    filters_tensor = np.random.rand(1, 1, 1, 1).astype(np.float32)
    output_shape_tensor = np.array([1, 3, 3, 1], dtype=np.int32)
    strides_int = 1
    padding_str = 'VALID'
    data_format_str = 'NHWC'
    dilations_int = 1
    name_str = 'transpose_conv5'

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "output_shape": output_shape_tensor,
        "strides": strides_int,
        "padding": padding_str,
        "data_format": data_format_str,
        "dilations": dilations_int,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.rand(1, 5, 5, 2).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 2, 2).astype(np.float32)
    output_shape_tensor = np.array([1, 7, 7, 2], dtype=np.int32)
    strides_int = 1
    padding_str = 'SAME'
    data_format_str = 'NHWC'
    dilations_int = 1
    name_str = None

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "output_shape": output_shape_tensor,
        "strides": strides_int,
        "padding": padding_str,
        "data_format": data_format_str,
        "dilations": dilations_int,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.random.rand(2, 5, 5, 2).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 2, 2).astype(np.float32)
    output_shape_tensor = np.array([2, 9, 9, 2], dtype=np.int32)
    strides_int = 2
    padding_str = 'SAME'
    data_format_str = 'NHWC'
    dilations_int = 1
    name_str = 'transpose_conv7'

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "output_shape": output_shape_tensor,
        "strides": strides_int,
        "padding": padding_str,
        "data_format": data_format_str,
        "dilations": dilations_int,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.random.rand(1, 10, 10, 1).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 1, 2).astype(np.float32)
    output_shape_tensor = np.array([1, 12, 12, 2], dtype=np.int32)
    strides_int = 1
    padding_str = 'VALID'
    data_format_str = 'NHWC'
    dilations_int = 1
    name_str = 'transpose_conv8'

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "output_shape": output_shape_tensor,
        "strides": strides_int,
        "padding": padding_str,
        "data_format": data_format_str,
        "dilations": dilations_int,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(1, 4, 4, 2).astype(np.float32)
    filters_tensor = np.random.rand(2, 2, 2, 2).astype(np.float32)
    output_shape_tensor = np.array([1, 6, 6, 2], dtype=np.int32)
    strides_int = 1
    padding_str = 'SAME'
    data_format_str = 'NHWC'
    dilations_int = 1
    name_str = 'transpose_conv9'

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "output_shape": output_shape_tensor,
        "strides": strides_int,
        "padding": padding_str,
        "data_format": data_format_str,
        "dilations": dilations_int,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.random.rand(1, 3, 3, 1).astype(np.float32)
    filters_tensor = np.random.rand(2, 2, 1, 1).astype(np.float32)
    output_shape_tensor = np.array([1, 5, 5, 1], dtype=np.int32)
    strides_int = 2
    padding_str = 'VALID'
    data_format_str = 'NHWC'
    dilations_int = 1
    name_str = 'transpose_conv10'

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "output_shape": output_shape_tensor,
        "strides": strides_int,
        "padding": padding_str,
        "data_format": data_format_str,
        "dilations": dilations_int,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.conv_transpose_1"] = tf_nn_conv_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.conv_transpose_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.conv_transpose_1'.")

check_valid('tf.nn.conv_transpose', generated_inputs['tf.nn.conv_transpose_1'], lib="tf", suffix=1)
