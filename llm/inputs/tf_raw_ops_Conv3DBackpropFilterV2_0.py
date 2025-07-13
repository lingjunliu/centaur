
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Conv3DBackpropFilterV2_inputs():
    list_of_inputs = []

    # Input 1
    input_val = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    filter_sizes_val = np.array([3, 3, 3, 3, 2], dtype=np.int32)
    out_backprop_val = np.random.rand(1, 3, 3, 3, 2).astype(np.float32)
    strides_val = [1, 1, 1, 1, 1]
    padding_val = "VALID"
    data_format_val = "NDHWC"
    dilations_val = [1, 1, 1, 1, 1]
    name_val = "conv3d_backprop_filter_v2_1"

    input_dict = {
        "input": input_val,
        "filter_sizes": filter_sizes_val,
        "out_backprop": out_backprop_val,
        "strides": strides_val,
        "padding": padding_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_val = np.random.rand(2, 7, 7, 7, 5).astype(np.float32)
    filter_sizes_val = np.array([2, 2, 2, 5, 4], dtype=np.int32)
    out_backprop_val = np.random.rand(2, 6, 6, 6, 4).astype(np.float32)
    strides_val = [1, 1, 1, 1, 1]
    padding_val = "VALID"
    data_format_val = "NDHWC"
    dilations_val = [1, 1, 1, 1, 1]
    name_val = "conv3d_backprop_filter_v2_2"

    input_dict = {
        "input": input_val,
        "filter_sizes": filter_sizes_val,
        "out_backprop": out_backprop_val,
        "strides": strides_val,
        "padding": padding_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_val = np.random.rand(1, 8, 8, 8, 3).astype(np.float32)
    filter_sizes_val = np.array([4, 4, 4, 3, 2], dtype=np.int32)
    out_backprop_val = np.random.rand(1, 8, 8, 8, 2).astype(np.float32)
    strides_val = [1, 1, 1, 1, 1]
    padding_val = "SAME"
    data_format_val = "NDHWC"
    dilations_val = [1, 1, 1, 1, 1]
    name_val = "conv3d_backprop_filter_v2_3"

    input_dict = {
        "input": input_val,
        "filter_sizes": filter_sizes_val,
        "out_backprop": out_backprop_val,
        "strides": strides_val,
        "padding": padding_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4
    input_val = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    filter_sizes_val = np.array([3, 3, 3, 3, 2], dtype=np.int32)
    out_backprop_val = np.random.rand(1, 5, 5, 5, 2).astype(np.float32)
    strides_val = [1, 1, 1, 1, 1]
    padding_val = "SAME"
    data_format_val = "NDHWC"
    dilations_val = [1, 1, 1, 1, 1]
    name_val = "conv3d_backprop_filter_v2_4"

    input_dict = {
        "input": input_val,
        "filter_sizes": filter_sizes_val,
        "out_backprop": out_backprop_val,
        "strides": strides_val,
        "padding": padding_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: NCDHW data format
    input_val = np.random.rand(1, 3, 5, 5, 5).astype(np.float32)
    filter_sizes_val = np.array([3, 3, 3, 3, 2], dtype=np.int32)
    out_backprop_val = np.random.rand(1, 2, 3, 3, 3).astype(np.float32)
    strides_val = [1, 1, 1, 1, 1]
    padding_val = "VALID"
    data_format_val = "NCDHW"
    dilations_val = [1, 1, 1, 1, 1]
    name_val = "conv3d_backprop_filter_v2_5"

    input_dict = {
        "input": input_val,
        "filter_sizes": filter_sizes_val,
        "out_backprop": out_backprop_val,
        "strides": strides_val,
        "padding": padding_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: different strides
    input_val = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    filter_sizes_val = np.array([3, 3, 3, 3, 2], dtype=np.int32)
    out_backprop_val = np.random.rand(1, 2, 2, 2, 2).astype(np.float32)
    strides_val = [1, 1, 2, 2, 1]
    padding_val = "VALID"
    data_format_val = "NDHWC"
    dilations_val = [1, 1, 1, 1, 1]
    name_val = "conv3d_backprop_filter_v2_6"

    input_dict = {
        "input": input_val,
        "filter_sizes": filter_sizes_val,
        "out_backprop": out_backprop_val,
        "strides": strides_val,
        "padding": padding_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different dilations
    input_val = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    filter_sizes_val = np.array([3, 3, 3, 3, 2], dtype=np.int32)
    out_backprop_val = np.random.rand(1, 1, 1, 1, 2).astype(np.float32)
    strides_val = [1, 1, 1, 1, 1]
    padding_val = "VALID"
    data_format_val = "NDHWC"
    dilations_val = [1, 1, 2, 2, 1]
    name_val = "conv3d_backprop_filter_v2_7"

    input_dict = {
        "input": input_val,
        "filter_sizes": filter_sizes_val,
        "out_backprop": out_backprop_val,
        "strides": strides_val,
        "padding": padding_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float16 type
    input_val = np.random.rand(1, 5, 5, 5, 3).astype(np.float16)
    filter_sizes_val = np.array([3, 3, 3, 3, 2], dtype=np.int32)
    out_backprop_val = np.random.rand(1, 3, 3, 3, 2).astype(np.float16)
    strides_val = [1, 1, 1, 1, 1]
    padding_val = "VALID"
    data_format_val = "NDHWC"
    dilations_val = [1, 1, 1, 1, 1]
    name_val = "conv3d_backprop_filter_v2_8"

    input_dict = {
        "input": input_val,
        "filter_sizes": filter_sizes_val,
        "out_backprop": out_backprop_val,
        "strides": strides_val,
        "padding": padding_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float16 type
    input_val = np.random.rand(1, 5, 5, 5, 3).astype(np.float16)
    filter_sizes_val = np.array([3, 3, 3, 3, 2], dtype=np.int32)
    out_backprop_val = np.random.rand(1, 3, 3, 3, 2).astype(np.float16)
    strides_val = [1, 1, 1, 1, 1]
    padding_val = "VALID"
    data_format_val = "NDHWC"
    dilations_val = [1, 1, 1, 1, 1]
    name_val = "conv3d_backprop_filter_v2_9"

    input_dict = {
        "input": input_val,
        "filter_sizes": filter_sizes_val,
        "out_backprop": out_backprop_val,
        "strides": strides_val,
        "padding": padding_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64 type
    input_val = np.random.rand(1, 5, 5, 5, 3).astype(np.float64)
    filter_sizes_val = np.array([3, 3, 3, 3, 2], dtype=np.int32)
    out_backprop_val = np.random.rand(1, 3, 3, 3, 2).astype(np.float64)
    strides_val = [1, 1, 1, 1, 1]
    padding_val = "VALID"
    data_format_val = "NDHWC"
    dilations_val = [1, 1, 1, 1, 1]
    name_val = "conv3d_backprop_filter_v2_10"

    input_dict = {
        "input": input_val,
        "filter_sizes": filter_sizes_val,
        "out_backprop": out_backprop_val,
        "strides": strides_val,
        "padding": padding_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
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
