
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Conv3DBackpropInputV2_inputs():
    list_of_inputs = []

    # Input 1
    input_sizes = np.array([1, 3, 10, 10, 3], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 3, 5).astype(np.float32)
    out_backprop = np.random.rand(1, 1, 8, 8, 5).astype(np.float32)
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    dilations = [1, 1, 1, 1, 1]
    name = "conv3d_backprop_input_1"

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_sizes = np.array([2, 5, 12, 12, 1], dtype=np.int64)
    filter_val = np.random.rand(2, 2, 2, 1, 4).astype(np.float64)
    out_backprop = np.random.rand(2, 4, 11, 11, 4).astype(np.float64)
    strides = [1, 2, 1, 1, 1]
    padding = "SAME"
    data_format = "NCDHW"
    dilations = [1, 1, 2, 2, 1]
    name = "conv3d_backprop_input_2"

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_sizes = np.array([4, 7, 15, 15, 8], dtype=np.int32)
    filter_val = np.random.rand(4, 4, 4, 8, 2).astype(np.float32)
    out_backprop = np.random.rand(4, 4, 12, 12, 2).astype(np.float32)
    strides = [1, 1, 2, 2, 1]
    padding = "VALID"
    data_format = "NDHWC"
    dilations = [1, 2, 1, 1, 1]
    name = "conv3d_backprop_input_3"

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_sizes = np.array([1, 9, 18, 18, 16], dtype=np.int64)
    filter_val = np.random.rand(5, 5, 5, 16, 1).astype(np.float64)
    out_backprop = np.random.rand(1, 5, 14, 14, 1).astype(np.float64)
    strides = [1, 3, 1, 1, 1]
    padding = "SAME"
    data_format = "NCDHW"
    dilations = [1, 1, 3, 1, 1]
    name = "conv3d_backprop_input_4"

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_sizes = np.array([2, 11, 20, 20, 32], dtype=np.int32)
    filter_val = np.random.rand(6, 6, 6, 32, 8).astype(np.float32)
    out_backprop = np.random.rand(2, 6, 15, 15, 8).astype(np.float32)
    strides = [1, 1, 3, 3, 1]
    padding = "VALID"
    data_format = "NDHWC"
    dilations = [1, 3, 3, 1, 1]
    name = "conv3d_backprop_input_5"

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_sizes = np.array([1, 3, 10, 10, 3], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 3, 5).astype(np.float32)
    out_backprop = np.random.rand(1, 1, 8, 8, 5).astype(np.float32)
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    dilations = [1, 1, 1, 1, 1]
    name = None

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_sizes = np.array([1, 3, 10, 10, 3], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 3, 5).astype(np.float32)
    out_backprop = np.random.rand(1, 1, 8, 8, 5).astype(np.float32)
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    data_format = "NDHWC"
    dilations = [1, 1, 1, 1, 1]
    name = "conv3d_backprop_input_7"

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_sizes = np.array([1, 3, 10, 10, 3], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 3, 5).astype(np.float32)
    out_backprop = np.random.rand(1, 1, 8, 8, 5).astype(np.float32)
    strides = [1, 2, 2, 2, 1]
    padding = "VALID"
    data_format = "NDHWC"
    dilations = [1, 1, 1, 1, 1]
    name = "conv3d_backprop_input_8"

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_sizes = np.array([1, 3, 10, 10, 3], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 3, 5).astype(np.float32)
    out_backprop = np.random.rand(1, 1, 8, 8, 5).astype(np.float32)
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    dilations = [1, 2, 2, 2, 1]
    name = "conv3d_backprop_input_9"

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_sizes = np.array([1, 3, 10, 10, 3], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 3, 5).astype(np.float32)
    out_backprop = np.random.rand(1, 1, 8, 8, 5).astype(np.float32)
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NCDHW"
    dilations = [1, 1, 1, 1, 1]
    name = "conv3d_backprop_input_10"

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
temp_inputs = tf_raw_ops_Conv3DBackpropInputV2_inputs()
for input_dict in temp_inputs:
    input_dict["input_sizes"] = tf.convert_to_tensor(input_dict["input_sizes"])
    input_dict["filter"] = tf.convert_to_tensor(input_dict["filter"])
    input_dict["out_backprop"] = tf.convert_to_tensor(input_dict["out_backprop"])
    input_dict["strides"] = input_dict["strides"]
    input_dict["padding"] = input_dict["padding"]
    input_dict["data_format"] = input_dict["data_format"]
    input_dict["dilations"] = input_dict["dilations"]
    input_dict["name"] = input_dict["name"]

generated_inputs["tf.raw_ops.Conv3DBackpropInputV2"] = temp_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Conv3DBackpropInputV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Conv3DBackpropInputV2'.")

check_valid('tf.raw_ops.Conv3DBackpropInputV2', generated_inputs['tf.raw_ops.Conv3DBackpropInputV2'], lib="tf", suffix=0)
