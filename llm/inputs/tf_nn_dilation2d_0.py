
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_dilation2d_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    input1 = np.random.rand(1, 10, 10, 3).astype(np.float32)
    filters1 = np.random.rand(3, 3, 3).astype(np.float32)
    strides1 = [1, 1, 1, 1]
    padding1 = "VALID"
    data_format1 = "NHWC"
    dilations1 = [1, 1, 1, 1]
    name1 = "dilation1"

    input_dict1 = {
        "input": input1,
        "filters": filters1,
        "strides": strides1,
        "padding": padding1,
        "data_format": data_format1,
        "dilations": dilations1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: SAME padding
    input2 = np.random.rand(1, 10, 10, 3).astype(np.float32)
    filters2 = np.random.rand(3, 3, 3).astype(np.float32)
    strides2 = [1, 1, 1, 1]
    padding2 = "SAME"
    data_format2 = "NHWC"
    dilations2 = [1, 1, 1, 1]
    name2 = "dilation2"

    input_dict2 = {
        "input": input2,
        "filters": filters2,
        "strides": strides2,
        "padding": padding2,
        "data_format": data_format2,
        "dilations": dilations2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Different strides
    input3 = np.random.rand(1, 10, 10, 3).astype(np.float32)
    filters3 = np.random.rand(3, 3, 3).astype(np.float32)
    strides3 = [1, 2, 2, 1]
    padding3 = "VALID"
    data_format3 = "NHWC"
    dilations3 = [1, 1, 1, 1]
    name3 = "dilation3"

    input_dict3 = {
        "input": input3,
        "filters": filters3,
        "strides": strides3,
        "padding": padding3,
        "data_format": data_format3,
        "dilations": dilations3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Different dilations
    input4 = np.random.rand(1, 10, 10, 3).astype(np.float32)
    filters4 = np.random.rand(3, 3, 3).astype(np.float32)
    strides4 = [1, 1, 1, 1]
    padding4 = "VALID"
    data_format4 = "NHWC"
    dilations4 = [1, 2, 2, 1]
    name4 = "dilation4"

    input_dict4 = {
        "input": input4,
        "filters": filters4,
        "strides": strides4,
        "padding": padding4,
        "data_format": data_format4,
        "dilations": dilations4,
        "name": name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Different input size
    input5 = np.random.rand(1, 15, 15, 3).astype(np.float32)
    filters5 = np.random.rand(3, 3, 3).astype(np.float32)
    strides5 = [1, 1, 1, 1]
    padding5 = "VALID"
    data_format5 = "NHWC"
    dilations5 = [1, 1, 1, 1]
    name5 = "dilation5"

    input_dict5 = {
        "input": input5,
        "filters": filters5,
        "strides": strides5,
        "padding": padding5,
        "data_format": data_format5,
        "dilations": dilations5,
        "name": name5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

     # Input 6: Different filter size
    input6 = np.random.rand(1, 10, 10, 3).astype(np.float32)
    filters6 = np.random.rand(5, 5, 3).astype(np.float32)
    strides6 = [1, 1, 1, 1]
    padding6 = "VALID"
    data_format6 = "NHWC"
    dilations6 = [1, 1, 1, 1]
    name6 = "dilation6"

    input_dict6 = {
        "input": input6,
        "filters": filters6,
        "strides": strides6,
        "padding": padding6,
        "data_format": data_format6,
        "dilations": dilations6,
        "name": name6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: batch size = 2
    input7 = np.random.rand(2, 10, 10, 3).astype(np.float32)
    filters7 = np.random.rand(3, 3, 3).astype(np.float32)
    strides7 = [1, 1, 1, 1]
    padding7 = "VALID"
    data_format7 = "NHWC"
    dilations7 = [1, 1, 1, 1]
    name7 = "dilation7"

    input_dict7 = {
        "input": input7,
        "filters": filters7,
        "strides": strides7,
        "padding": padding7,
        "data_format": data_format7,
        "dilations": dilations7,
        "name": name7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: different depth
    input8 = np.random.rand(1, 10, 10, 5).astype(np.float32)
    filters8 = np.random.rand(3, 3, 5).astype(np.float32)
    strides8 = [1, 1, 1, 1]
    padding8 = "VALID"
    data_format8 = "NHWC"
    dilations8 = [1, 1, 1, 1]
    name8 = "dilation8"

    input_dict8 = {
        "input": input8,
        "filters": filters8,
        "strides": strides8,
        "padding": padding8,
        "data_format": data_format8,
        "dilations": dilations8,
        "name": name8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Integer type
    input9 = np.random.randint(0, 10, size=(1, 10, 10, 3)).astype(np.int32)
    filters9 = np.random.randint(0, 10, size=(3, 3, 3)).astype(np.int32)
    strides9 = [1, 1, 1, 1]
    padding9 = "VALID"
    data_format9 = "NHWC"
    dilations9 = [1, 1, 1, 1]
    name9 = "dilation9"

    input_dict9 = {
        "input": input9,
        "filters": filters9,
        "strides": strides9,
        "padding": padding9,
        "data_format": data_format9,
        "dilations": dilations9,
        "name": name9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: uint8 type
    input10 = np.random.randint(0, 255, size=(1, 10, 10, 3)).astype(np.uint8)
    filters10 = np.random.randint(0, 255, size=(3, 3, 3)).astype(np.uint8)
    strides10 = [1, 1, 1, 1]
    padding10 = "VALID"
    data_format10 = "NHWC"
    dilations10 = [1, 1, 1, 1]
    name10 = "dilation10"

    input_dict10 = {
        "input": input10,
        "filters": filters10,
        "strides": strides10,
        "padding": padding10,
        "data_format": data_format10,
        "dilations": dilations10,
        "name": name10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.dilation2d"] = tf_nn_dilation2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.dilation2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.dilation2d'.")

check_valid('tf.nn.dilation2d', generated_inputs['tf.nn.dilation2d'], lib="tf", suffix=0)
