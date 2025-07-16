
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DepthwiseConv2dNativeBackpropInput_inputs():
    list_of_inputs = []

    # Input 1
    input_sizes = np.array([1, 10, 10, 3], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 2).astype(np.float32)
    out_backprop = np.random.rand(1, 8, 8, 6).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "VALID"
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = None

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_sizes = np.array([2, 20, 20, 5], dtype=np.int32)
    filter_val = np.random.rand(5, 5, 5, 1).astype(np.float64)
    out_backprop = np.random.rand(2, 16, 16, 5).astype(np.float64)
    strides = [1, 1, 1, 1]
    padding = "SAME"
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "backprop_input_2"

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_sizes = np.array([1, 10, 10, 3], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 2).astype(np.float32)
    out_backprop = np.random.rand(1, 5, 5, 6).astype(np.float32)
    strides = [1, 2, 2, 1]
    padding = "VALID"
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = None

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_sizes = np.array([1, 10, 10, 3], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 2).astype(np.float16)
    out_backprop = np.random.rand(1, 10, 10, 6).astype(np.float16)
    strides = [1, 1, 1, 1]
    padding = "SAME"
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 2, 2, 1]
    name = None

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_sizes = np.array([1, 10, 10, 3], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 2).astype(np.float32)
    out_backprop = np.random.rand(1, 6, 6, 6).astype(np.float32)
    strides = [1, 2, 2, 1]
    padding = "VALID"
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = None

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_sizes = np.array([1, 10, 10, 3], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 2).astype(np.float32)
    out_backprop = np.random.rand(1, 10, 10, 6).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "EXPLICIT"
    explicit_paddings = [0, 1, 0, 1, 0, 1, 0, 1]
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = None

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    input_sizes = np.array([1, 10, 10, 3], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 2).astype(np.float32)
    out_backprop = np.random.rand(1, 12, 12, 6).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "SAME"
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 2, 2, 1]
    name = "dilation_test"

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_sizes = np.array([2, 15, 15, 4], dtype=np.int32)
    filter_val = np.random.rand(4, 4, 4, 3).astype(np.float64)
    out_backprop = np.random.rand(2, 8, 8, 12).astype(np.float64)
    strides = [1, 2, 2, 1]
    padding = "VALID"
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "valid_test"

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_sizes = np.array([1, 28, 28, 32], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 32, 1).astype(np.float32)
    out_backprop = np.random.rand(1, 28, 28, 32).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "SAME"
    explicit_paddings = []
    data_format = "NCHW"
    dilations = [1, 1, 1, 1]
    name = "nchw_test"

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_sizes = np.array([4, 8, 8, 16], dtype=np.int32)
    filter_val = np.random.rand(2, 2, 16, 2).astype(np.float32)
    out_backprop = np.random.rand(4, 4, 4, 32).astype(np.float32)
    strides = [1, 2, 2, 1]
    padding = "VALID"
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "large_batch_valid"

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DepthwiseConv2dNativeBackpropInput"] = tf_raw_ops_DepthwiseConv2dNativeBackpropInput_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DepthwiseConv2dNativeBackpropInput' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DepthwiseConv2dNativeBackpropInput'.")

check_valid('tf.raw_ops.DepthwiseConv2dNativeBackpropInput', generated_inputs['tf.raw_ops.DepthwiseConv2dNativeBackpropInput'], lib="tf", suffix=0)
