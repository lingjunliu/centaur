
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_depthwise_conv2d_backprop_input_inputs():
    list_of_inputs = []

    # Input 1
    input_sizes = np.array([1, 10, 10, 3], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 1).astype(np.float32)
    out_backprop = np.random.rand(1, 10, 10, 3).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "SAME"
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "depthwise_conv2d_backprop_input_1"

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
    input_sizes = np.array([2, 20, 20, 1], dtype=np.int32)
    filter_val = np.random.rand(5, 5, 1, 1).astype(np.float32)
    out_backprop = np.random.rand(2, 20, 20, 1).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "SAME"
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "depthwise_conv2d_backprop_input_2"

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
    input_sizes = np.array([1, 15, 15, 2], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 2, 1).astype(np.float32)
    out_backprop = np.random.rand(1, 15, 15, 2).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "SAME"
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "depthwise_conv2d_backprop_input_3"

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
    input_sizes = np.array([1, 30, 30, 4], dtype=np.int32)
    filter_val = np.random.rand(4, 4, 4, 1).astype(np.float32)
    out_backprop = np.random.rand(1, 30, 30, 4).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "SAME"
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "depthwise_conv2d_backprop_input_4"

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
    input_sizes = np.array([2, 16, 16, 3], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 1).astype(np.float32)
    out_backprop = np.random.rand(2, 16, 16, 3).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "SAME"
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "depthwise_conv2d_backprop_input_5"

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
    input_sizes = np.array([1, 8, 8, 4], dtype=np.int32)
    filter_val = np.random.rand(2, 2, 4, 1).astype(np.float32)
    out_backprop = np.random.rand(1, 8, 8, 4).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "SAME"
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "depthwise_conv2d_backprop_input_6"

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
generated_inputs["tf.nn.depthwise_conv2d_backprop_input"] = tf_nn_depthwise_conv2d_backprop_input_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.depthwise_conv2d_backprop_input' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.depthwise_conv2d_backprop_input'.")

check_valid('tf.nn.depthwise_conv2d_backprop_input', generated_inputs['tf.nn.depthwise_conv2d_backprop_input'], lib="tf", suffix=0)
