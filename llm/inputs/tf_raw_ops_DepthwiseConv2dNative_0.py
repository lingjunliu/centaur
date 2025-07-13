
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DepthwiseConv2dNative_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 32, 32, 3).astype(np.float32)
    filter_tensor = np.random.rand(3, 3, 3, 2).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "VALID"
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "depthwise_conv1"
    input_dict = {"input": input_tensor, "filter": filter_tensor, "strides": strides, "padding": padding, "explicit_paddings": explicit_paddings, "data_format": data_format, "dilations": dilations, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(1, 64, 64, 1).astype(np.float32)
    filter_tensor = np.random.rand(5, 5, 1, 4).astype(np.float32)
    strides = [1, 2, 2, 1]
    padding = "SAME"
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "depthwise_conv2"
    input_dict = {"input": input_tensor, "filter": filter_tensor, "strides": strides, "padding": padding, "explicit_paddings": explicit_paddings, "data_format": data_format, "dilations": dilations, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(2, 16, 16, 8).astype(np.float32)
    filter_tensor = np.random.rand(3, 3, 8, 1).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "VALID"
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 2, 2, 1]
    name = "depthwise_conv3"
    input_dict = {"input": input_tensor, "filter": filter_tensor, "strides": strides, "padding": padding, "explicit_paddings": explicit_paddings, "data_format": data_format, "dilations": dilations, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4
    input_tensor = np.random.rand(2, 16, 16, 8).astype(np.float32)
    filter_tensor = np.random.rand(3, 3, 8, 1).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "EXPLICIT"
    explicit_paddings = [0, 0, 1, 1, 0, 0, 1, 1]
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "depthwise_conv4"
    input_dict = {"input": input_tensor, "filter": filter_tensor, "strides": strides, "padding": padding, "explicit_paddings": explicit_paddings, "data_format": data_format, "dilations": dilations, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(1, 32, 32, 3).astype(np.float16)
    filter_tensor = np.random.rand(3, 3, 3, 2).astype(np.float16)
    strides = [1, 1, 1, 1]
    padding = "VALID"
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "depthwise_conv5"
    input_dict = {"input": input_tensor, "filter": filter_tensor, "strides": strides, "padding": padding, "explicit_paddings": explicit_paddings, "data_format": data_format, "dilations": dilations, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.rand(1, 32, 32, 3).astype(np.float64)
    filter_tensor = np.random.rand(3, 3, 3, 2).astype(np.float64)
    strides = [1, 1, 1, 1]
    padding = "VALID"
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "depthwise_conv6"
    input_dict = {"input": input_tensor, "filter": filter_tensor, "strides": strides, "padding": padding, "explicit_paddings": explicit_paddings, "data_format": data_format, "dilations": dilations, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 NCHW
    input_tensor = np.random.rand(1, 3, 32, 32).astype(np.float32)
    filter_tensor = np.random.rand(3, 3, 3, 2).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "VALID"
    explicit_paddings = []
    data_format = "NCHW"
    dilations = [1, 1, 1, 1]
    name = "depthwise_conv7"
    input_dict = {"input": input_tensor, "filter": filter_tensor, "strides": strides, "padding": padding, "explicit_paddings": explicit_paddings, "data_format": data_format, "dilations": dilations, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.random.rand(4, 8, 8, 4).astype(np.float32)
    filter_tensor = np.random.rand(2, 2, 4, 2).astype(np.float32)
    strides = [1, 2, 2, 1]
    padding = "SAME"
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "depthwise_conv8"
    input_dict = {"input": input_tensor, "filter": filter_tensor, "strides": strides, "padding": padding, "explicit_paddings": explicit_paddings, "data_format": data_format, "dilations": dilations, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(1, 64, 64, 3).astype(np.float32)
    filter_tensor = np.random.rand(3, 3, 3, 2).astype(np.float32)
    strides = [1, 2, 2, 1]
    padding = "SAME"
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "depthwise_conv9"
    input_dict = {"input": input_tensor, "filter": filter_tensor, "strides": strides, "padding": padding, "explicit_paddings": explicit_paddings, "data_format": data_format, "dilations": dilations, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 Dilations
    input_tensor = np.random.rand(1, 32, 32, 3).astype(np.float32)
    filter_tensor = np.random.rand(3, 3, 3, 2).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "VALID"
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 2, 2, 1]
    name = "depthwise_conv10"
    input_dict = {"input": input_tensor, "filter": filter_tensor, "strides": strides, "padding": padding, "explicit_paddings": explicit_paddings, "data_format": data_format, "dilations": dilations, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DepthwiseConv2dNative"] = tf_raw_ops_DepthwiseConv2dNative_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DepthwiseConv2dNative' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DepthwiseConv2dNative'.")

check_valid('tf.raw_ops.DepthwiseConv2dNative', generated_inputs['tf.raw_ops.DepthwiseConv2dNative'], lib="tf", suffix=0)
