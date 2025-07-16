
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DepthwiseConv2dNative_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.random.rand(1, 32, 32, 3).astype(np.float32)
    filter1 = np.random.rand(3, 3, 3, 2).astype(np.float32)
    strides1 = [1, 1, 1, 1]
    padding1 = "VALID"
    explicit_paddings1 = []
    data_format1 = "NHWC"
    dilations1 = [1, 1, 1, 1]
    name1 = "depthwise_conv1"

    input_dict1 = {
        "input": input1,
        "filter": filter1,
        "strides": strides1,
        "padding": padding1,
        "explicit_paddings": explicit_paddings1,
        "data_format": data_format1,
        "dilations": dilations1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.random.rand(4, 16, 16, 16).astype(np.float32)
    filter2 = np.random.rand(5, 5, 16, 4).astype(np.float32)
    strides2 = [1, 2, 2, 1]
    padding2 = "SAME"
    explicit_paddings2 = []
    data_format2 = "NHWC"
    dilations2 = [1, 1, 1, 1]
    name2 = "depthwise_conv2"

    input_dict2 = {
        "input": input2,
        "filter": filter2,
        "strides": strides2,
        "padding": padding2,
        "explicit_paddings": explicit_paddings2,
        "data_format": data_format2,
        "dilations": dilations2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = np.random.rand(1, 64, 64, 1).astype(np.float32)
    filter3 = np.random.rand(7, 7, 1, 8).astype(np.float32)
    strides3 = [1, 4, 4, 1]
    padding3 = "VALID"
    explicit_paddings3 = []
    data_format3 = "NHWC"
    dilations3 = [1, 1, 1, 1]
    name3 = "depthwise_conv3"

    input_dict3 = {
        "input": input3,
        "filter": filter3,
        "strides": strides3,
        "padding": padding3,
        "explicit_paddings": explicit_paddings3,
        "data_format": data_format3,
        "dilations": dilations3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    input4 = np.random.rand(2, 8, 8, 32).astype(np.float32)
    filter4 = np.random.rand(1, 1, 32, 1).astype(np.float32)
    strides4 = [1, 1, 1, 1]
    padding4 = "SAME"
    explicit_paddings4 = []
    data_format4 = "NHWC"
    dilations4 = [1, 2, 2, 1]
    name4 = "depthwise_conv4"

    input_dict4 = {
        "input": input4,
        "filter": filter4,
        "strides": strides4,
        "padding": padding4,
        "explicit_paddings": explicit_paddings4,
        "data_format": data_format4,
        "dilations": dilations4,
        "name": name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    input5 = np.random.rand(1, 10, 10, 64).astype(np.float32)
    filter5 = np.random.rand(3, 3, 64, 2).astype(np.float32)
    strides5 = [1, 1, 1, 1]
    padding5 = "EXPLICIT"
    explicit_paddings5 = [0, 0, 1, 1, 1, 1, 0, 0]
    data_format5 = "NHWC"
    dilations5 = [1, 1, 1, 1]
    name5 = "depthwise_conv5"

    input_dict5 = {
        "input": input5,
        "filter": filter5,
        "strides": strides5,
        "padding": padding5,
        "explicit_paddings": explicit_paddings5,
        "data_format": data_format5,
        "dilations": dilations5,
        "name": name5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6
    input6 = np.random.rand(1, 10, 10, 64).astype(np.float32)
    filter6 = np.random.rand(3, 3, 64, 2).astype(np.float32)
    strides6 = [1, 1, 1, 1]
    padding6 = "SAME"
    explicit_paddings6 = []
    data_format6 = "NCHW"
    dilations6 = [1, 1, 1, 1]
    name6 = "depthwise_conv6"

    input_dict6 = {
        "input": input6,
        "filter": filter6,
        "strides": strides6,
        "padding": padding6,
        "explicit_paddings": explicit_paddings6,
        "data_format": data_format6,
        "dilations": dilations6,
        "name": name6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

        # Input 7
    input7 = np.random.rand(2, 32, 32, 3).astype(np.float16)
    filter7 = np.random.rand(3, 3, 3, 2).astype(np.float16)
    strides7 = [1, 1, 1, 1]
    padding7 = "VALID"
    explicit_paddings7 = []
    data_format7 = "NHWC"
    dilations7 = [1, 1, 1, 1]
    name7 = "depthwise_conv7"

    input_dict7 = {
        "input": input7,
        "filter": filter7,
        "strides": strides7,
        "padding": padding7,
        "explicit_paddings": explicit_paddings7,
        "data_format": data_format7,
        "dilations": dilations7,
        "name": name7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    input8 = np.random.rand(1, 64, 64, 1).astype(np.float64)
    filter8 = np.random.rand(7, 7, 1, 8).astype(np.float64)
    strides8 = [1, 4, 4, 1]
    padding8 = "VALID"
    explicit_paddings8 = []
    data_format8 = "NHWC"
    dilations8 = [1, 1, 1, 1]
    name8 = "depthwise_conv8"

    input_dict8 = {
        "input": input8,
        "filter": filter8,
        "strides": strides8,
        "padding": padding8,
        "explicit_paddings": explicit_paddings8,
        "data_format": data_format8,
        "dilations": dilations8,
        "name": name8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Input 9, removed bfloat16 because it's not available in numpy
    input9 = np.random.rand(1, 64, 64, 1).astype(np.float32)
    filter9 = np.random.rand(7, 7, 1, 8).astype(np.float32)
    strides9 = [1, 4, 4, 1]
    padding9 = "VALID"
    explicit_paddings9 = []
    data_format9 = "NHWC"
    dilations9 = [1, 1, 1, 1]
    name9 = "depthwise_conv9"

    input_dict9 = {
        "input": input9,
        "filter": filter9,
        "strides": strides9,
        "padding": padding9,
        "explicit_paddings": explicit_paddings9,
        "data_format": data_format9,
        "dilations": dilations9,
        "name": name9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    # Input 10
    input10 = np.random.rand(4, 16, 16, 16).astype(np.float64)
    filter10 = np.random.rand(5, 5, 16, 4).astype(np.float64)
    strides10 = [1, 2, 2, 1]
    padding10 = "SAME"
    explicit_paddings10 = []
    data_format10 = "NHWC"
    dilations10 = [1, 1, 1, 1]
    name10 = "depthwise_conv10"

    input_dict10 = {
        "input": input10,
        "filter": filter10,
        "strides": strides10,
        "padding": padding10,
        "explicit_paddings": explicit_paddings10,
        "data_format": data_format10,
        "dilations": dilations10,
        "name": name10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    

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
