
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Conv3DBackpropFilterV2_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    filter_sizes_tensor = np.array([3, 3, 3, 3, 2], dtype=np.int32)
    out_backprop_tensor = np.random.rand(1, 3, 3, 3, 2).astype(np.float32)
    strides_list = [1, 1, 1, 1, 1]
    padding_string = "VALID"
    data_format_string = "NDHWC"
    dilations_list = [1, 1, 1, 1, 1]
    name_string = "conv3d_backprop_filter_1"

    input_dict = {
        "input": input_tensor,
        "filter_sizes": filter_sizes_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(2, 10, 10, 10, 5).astype(np.float64)
    filter_sizes_tensor = np.array([5, 5, 5, 5, 4], dtype=np.int32)
    out_backprop_tensor = np.random.rand(2, 3, 3, 3, 4).astype(np.float64)
    strides_list = [1, 2, 2, 2, 1]
    padding_string = "VALID"
    data_format_string = "NDHWC"
    dilations_list = [1, 1, 1, 1, 1]
    name_string = "conv3d_backprop_filter_2"

    input_dict = {
        "input": input_tensor,
        "filter_sizes": filter_sizes_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(1, 8, 8, 8, 3).astype(np.float32)
    filter_sizes_tensor = np.array([2, 2, 2, 3, 2], dtype=np.int32)
    out_backprop_tensor = np.random.rand(1, 7, 7, 7, 2).astype(np.float32)
    strides_list = [1, 1, 1, 1, 1]
    padding_string = "VALID"
    data_format_string = "NDHWC"
    dilations_list = [1, 1, 1, 1, 1]
    name_string = "conv3d_backprop_filter_3"

    input_dict = {
        "input": input_tensor,
        "filter_sizes": filter_sizes_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(4, 12, 12, 12, 4).astype(np.float16)
    filter_sizes_tensor = np.array([4, 4, 4, 4, 3], dtype=np.int32)
    out_backprop_tensor = np.random.rand(4, 9, 9, 9, 3).astype(np.float16)
    strides_list = [1, 1, 1, 1, 1]
    padding_string = "VALID"
    data_format_string = "NDHWC"
    dilations_list = [1, 1, 1, 1, 1]
    name_string = "conv3d_backprop_filter_4"

    input_dict = {
        "input": input_tensor,
        "filter_sizes": filter_sizes_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    input_tensor = np.random.rand(1, 7, 7, 7, 1).astype(np.float16)
    filter_sizes_tensor = np.array([3, 3, 3, 1, 1], dtype=np.int32)
    out_backprop_tensor = np.random.rand(1, 5, 5, 5, 1).astype(np.float16)
    strides_list = [1, 1, 1, 1, 1]
    padding_string = "VALID"
    data_format_string = "NDHWC"
    dilations_list = [1, 1, 1, 1, 1]
    name_string = "conv3d_backprop_filter_5"

    input_dict = {
        "input": input_tensor,
        "filter_sizes": filter_sizes_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.rand(2, 6, 6, 6, 2).astype(np.float32)
    filter_sizes_tensor = np.array([2, 2, 2, 2, 1], dtype=np.int32)
    out_backprop_tensor = np.random.rand(2, 5, 5, 5, 1).astype(np.float32)
    strides_list = [1, 1, 1, 1, 1]
    padding_string = "VALID"
    data_format_string = "NDHWC"
    dilations_list = [1, 1, 1, 1, 1]
    name_string = "conv3d_backprop_filter_6"

    input_dict = {
        "input": input_tensor,
        "filter_sizes": filter_sizes_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.random.rand(1, 15, 15, 15, 6).astype(np.float64)
    filter_sizes_tensor = np.array([7, 7, 7, 6, 5], dtype=np.int32)
    out_backprop_tensor = np.random.rand(1, 4, 4, 4, 5).astype(np.float64)
    strides_list = [1, 2, 2, 2, 1]
    padding_string = "VALID"
    data_format_string = "NDHWC"
    dilations_list = [1, 1, 1, 1, 1]
    name_string = "conv3d_backprop_filter_7"

    input_dict = {
        "input": input_tensor,
        "filter_sizes": filter_sizes_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.random.rand(3, 9, 9, 9, 2).astype(np.float16)
    filter_sizes_tensor = np.array([3, 3, 3, 2, 3], dtype=np.int32)
    out_backprop_tensor = np.random.rand(3, 7, 7, 7, 3).astype(np.float16)
    strides_list = [1, 1, 1, 1, 1]
    padding_string = "VALID"
    data_format_string = "NDHWC"
    dilations_list = [1, 1, 1, 1, 1]
    name_string = "conv3d_backprop_filter_8"

    input_dict = {
        "input": input_tensor,
        "filter_sizes": filter_sizes_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(1, 4, 4, 4, 1).astype(np.float32)
    filter_sizes_tensor = np.array([2, 2, 2, 1, 1], dtype=np.int32)
    out_backprop_tensor = np.random.rand(1, 3, 3, 3, 1).astype(np.float32)
    strides_list = [1, 1, 1, 1, 1]
    padding_string = "VALID"
    data_format_string = "NDHWC"
    dilations_list = [1, 1, 1, 1, 1]
    name_string = "conv3d_backprop_filter_9"

    input_dict = {
        "input": input_tensor,
        "filter_sizes": filter_sizes_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10
    input_tensor = np.random.rand(1, 11, 11, 11, 3).astype(np.float16)
    filter_sizes_tensor = np.array([5, 5, 5, 3, 2], dtype=np.int32)
    out_backprop_tensor = np.random.rand(1, 7, 7, 7, 2).astype(np.float16)
    strides_list = [1, 1, 1, 1, 1]
    padding_string = "VALID"
    data_format_string = "NDHWC"
    dilations_list = [1, 1, 1, 1, 1]
    name_string = "conv3d_backprop_filter_10"

    input_dict = {
        "input": input_tensor,
        "filter_sizes": filter_sizes_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Conv3DBackpropFilterV2"] = tf_raw_ops_Conv3DBackpropFilterV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Conv3DBackpropFilterV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Conv3DBackpropFilterV2'.")

check_valid('tf.raw_ops.Conv3DBackpropFilterV2', generated_inputs['tf.raw_ops.Conv3DBackpropFilterV2'], lib="tf", suffix=0)
