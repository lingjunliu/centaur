
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_depthwise_conv2d_native_backprop_input_inputs():
    list_of_inputs = []

    # Input 1
    input_sizes = np.array([1, 5, 5, 3], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 1).astype(np.float32)
    out_backprop = np.random.rand(1, 3, 3, 3).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "VALID"
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_sizes = np.array([1, 7, 7, 3], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 1).astype(np.float32)
    out_backprop = np.random.rand(1, 5, 5, 3).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "SAME"
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DepthwiseConv2dNativeBackpropInput"] = tf_raw_ops_depthwise_conv2d_native_backprop_input_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DepthwiseConv2dNativeBackpropInput' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DepthwiseConv2dNativeBackpropInput'.")

check_valid('tf.raw_ops.DepthwiseConv2dNativeBackpropInput', generated_inputs['tf.raw_ops.DepthwiseConv2dNativeBackpropInput'], lib="tf", suffix=0)
