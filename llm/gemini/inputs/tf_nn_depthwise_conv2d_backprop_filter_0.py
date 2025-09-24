
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_depthwise_conv2d_backprop_filter_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.random.rand(1, 10, 10, 3).astype(np.float32)
    filter_sizes1 = np.array([3, 3, 3, 1], dtype=np.int32)
    out_backprop1 = np.random.rand(1, 8, 8, 3).astype(np.float32)
    strides1 = [1, 1, 1, 1]
    padding1 = "VALID"
    data_format1 = "NHWC"
    dilations1 = [1, 1, 1, 1]
    name1 = "depthwise_conv2d_backprop_filter_1"

    input_dict1 = {
        "input": input1,
        "filter_sizes": filter_sizes1,
        "out_backprop": out_backprop1,
        "strides": strides1,
        "padding": padding1,
        "data_format": data_format1,
        "dilations": dilations1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.random.rand(4, 20, 20, 5).astype(np.float32)
    filter_sizes2 = np.array([5, 5, 5, 2], dtype=np.int32)
    out_backprop2 = np.random.rand(4, 16, 16, 10).astype(np.float32)
    strides2 = [1, 1, 1, 1]
    padding2 = "VALID"
    data_format2 = "NHWC"
    dilations2 = [1, 1, 1, 1]
    name2 = "depthwise_conv2d_backprop_filter_2"

    input_dict2 = {
        "input": input2,
        "filter_sizes": filter_sizes2,
        "out_backprop": out_backprop2,
        "strides": strides2,
        "padding": padding2,
        "data_format": data_format2,
        "dilations": dilations2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = np.random.rand(1, 10, 10, 3).astype(np.float32)
    filter_sizes3 = np.array([3, 3, 3, 1], dtype=np.int32)
    out_backprop3 = np.random.rand(1, 10, 10, 3).astype(np.float32)
    strides3 = [1, 1, 1, 1]
    padding3 = "SAME"
    data_format3 = "NHWC"
    dilations3 = [1, 1, 1, 1]
    name3 = "depthwise_conv2d_backprop_filter_3"

    input_dict3 = {
        "input": input3,
        "filter_sizes": filter_sizes3,
        "out_backprop": out_backprop3,
        "strides": strides3,
        "padding": padding3,
        "data_format": data_format3,
        "dilations": dilations3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.depthwise_conv2d_backprop_filter"] = tf_nn_depthwise_conv2d_backprop_filter_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.depthwise_conv2d_backprop_filter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.depthwise_conv2d_backprop_filter'.")

check_valid('tf.nn.depthwise_conv2d_backprop_filter', generated_inputs['tf.nn.depthwise_conv2d_backprop_filter'], lib="tf", suffix=0)
