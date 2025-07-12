
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_conv3d_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    filters1 = np.random.rand(3, 3, 3, 3, 2).astype(np.float32)
    strides1 = [1, 1, 1, 1, 1]
    padding1 = "VALID"
    data_format1 = "NDHWC"
    dilations1 = [1, 1, 1, 1, 1]
    name1 = "conv3d_1"
    input_dict1 = {"input": input1, "filters": filters1, "strides": strides1, "padding": padding1, "data_format": data_format1, "dilations": dilations1, "name": name1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.random.rand(2, 10, 10, 10, 1).astype(np.float32)
    filters2 = np.random.rand(5, 5, 5, 1, 4).astype(np.float32)
    strides2 = [1, 2, 2, 2, 1]
    padding2 = "SAME"
    data_format2 = "NDHWC"
    dilations2 = [1, 1, 1, 1, 1]
    name2 = "conv3d_2"
    input_dict2 = {"input": input2, "filters": filters2, "strides": strides2, "padding": padding2, "data_format": data_format2, "dilations": dilations2, "name": name2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3 - Corrected in_channels
    input3 = np.random.rand(1, 8, 8, 8, 5).astype(np.float32)
    filters3 = np.random.rand(2, 2, 2, 5, 3).astype(np.float32)
    strides3 = [1, 1, 1, 1, 1]
    padding3 = "VALID"
    data_format3 = "NDHWC" # Changed to NDHWC to match filter shape
    dilations3 = [1, 1, 1, 1, 1]
    name3 = "conv3d_3"
    input_dict3 = {"input": input3, "filters": filters3, "strides": strides3, "padding": padding3, "data_format": data_format3, "dilations": dilations3, "name": name3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    input4 = np.random.rand(4, 6, 6, 6, 2).astype(np.float32)
    filters4 = np.random.rand(3, 3, 3, 2, 4).astype(np.float32)
    strides4 = [1, 2, 1, 2, 1]
    padding4 = "SAME"
    data_format4 = "NDHWC"
    dilations4 = [1, 2, 1, 2, 1]
    name4 = "conv3d_4"
    input_dict4 = {"input": input4, "filters": filters4, "strides": strides4, "padding": padding4, "data_format": data_format4, "dilations": dilations4, "name": name4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    input5 = np.random.rand(1, 7, 7, 7, 3).astype(np.float32)
    filters5 = np.random.rand(4, 4, 4, 3, 1).astype(np.float32)
    strides5 = [1, 1, 2, 1, 1]
    padding5 = "VALID"
    data_format5 = "NDHWC" # Changed to NDHWC to match filter shape
    dilations5 = [1, 1, 1, 1, 1]
    name5 = "conv3d_5"
    input_dict5 = {"input": input5, "filters": filters5, "strides": strides5, "padding": padding5, "data_format": data_format5, "dilations": dilations5, "name": name5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6
    input6 = np.random.rand(2, 9, 9, 9, 4).astype(np.float32)
    filters6 = np.random.rand(2, 2, 2, 4, 2).astype(np.float32)
    strides6 = [1, 3, 3, 3, 1]
    padding6 = "SAME"
    data_format6 = "NDHWC"
    dilations6 = [1, 1, 2, 1, 1]
    name6 = "conv3d_6"
    input_dict6 = {"input": input6, "filters": filters6, "strides": strides6, "padding": padding6, "data_format": data_format6, "dilations": dilations6, "name": name6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7
    input7 = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    filters7 = np.random.rand(3, 3, 3, 3, 2).astype(np.float32)
    strides7 = [1, 1, 1, 1, 1]
    padding7 = "VALID"
    data_format7 = "NDHWC"
    dilations7 = [1, 1, 1, 1, 1]
    name7 = "conv3d_7"
    input_dict7 = {"input": input7, "filters": filters7, "strides": strides7, "padding": padding7, "data_format": data_format7, "dilations": dilations7, "name": name7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    input8 = np.random.rand(2, 10, 10, 10, 1).astype(np.float32)
    filters8 = np.random.rand(5, 5, 5, 1, 4).astype(np.float32)
    strides8 = [1, 2, 2, 2, 1]
    padding8 = "SAME"
    data_format8 = "NDHWC"
    dilations8 = [1, 1, 1, 1, 1]
    name8 = "conv3d_8"
    input_dict8 = {"input": input8, "filters": filters8, "strides": strides8, "padding": padding8, "data_format": data_format8, "dilations": dilations8, "name": name8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9
    input9 = np.random.rand(1, 8, 8, 8, 5).astype(np.float32)
    filters9 = np.random.rand(2, 2, 2, 5, 3).astype(np.float32)
    strides9 = [1, 1, 1, 1, 1]
    padding9 = "VALID"
    data_format9 = "NDHWC"
    dilations9 = [1, 1, 1, 1, 1]
    name9 = "conv3d_9"
    input_dict9 = {"input": input9, "filters": filters9, "strides": strides9, "padding": padding9, "data_format": data_format9, "dilations": dilations9, "name": name9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10
    input10 = np.random.rand(4, 6, 6, 6, 2).astype(np.float32)
    filters10 = np.random.rand(3, 3, 3, 2, 4).astype(np.float32)
    strides10 = [1, 1, 1, 1, 1]
    padding10 = "SAME"
    data_format10 = "NDHWC"
    dilations10 = [1, 2, 2, 1, 1]
    name10 = "conv3d_10"
    input_dict10 = {"input": input10, "filters": filters10, "strides": strides10, "padding": padding10, "data_format": data_format10, "dilations": dilations10, "name": name10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.nn.conv3d"] = tf_nn_conv3d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.conv3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.conv3d'.")

check_valid('tf.nn.conv3d', generated_inputs['tf.nn.conv3d'], lib="tf", suffix=0)
