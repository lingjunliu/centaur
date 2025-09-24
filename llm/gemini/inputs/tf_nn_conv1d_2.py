
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_conv1d_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.random.rand(1, 10, 3).astype(np.float32)
    filters1 = np.random.rand(3, 3, 5).astype(np.float32)
    stride1 = [1]
    padding1 = 'VALID'
    data_format1 = 'NWC'
    dilations1 = [1]
    name1 = 'conv1d_1'

    input_dict1 = {
        'input': input1,
        'filters': filters1,
        'stride': stride1,
        'padding': padding1,
        'data_format': data_format1,
        'dilations': dilations1,
        'name': name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.random.rand(2, 20, 4).astype(np.float32)
    filters2 = np.random.rand(5, 4, 8).astype(np.float32)
    stride2 = [2]
    padding2 = 'SAME'
    data_format2 = 'NWC'
    dilations2 = [1]
    name2 = 'conv1d_2'

    input_dict2 = {
        'input': input2,
        'filters': filters2,
        'stride': stride2,
        'padding': padding2,
        'data_format': data_format2,
        'dilations': dilations2,
        'name': name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = np.random.rand(1, 15, 2).astype(np.float32)
    filters3 = np.random.rand(7, 2, 4).astype(np.float32)
    stride3 = [1]
    padding3 = 'VALID'
    data_format3 = 'NWC'
    dilations3 = [2]
    name3 = 'conv1d_3'

    input_dict3 = {
        'input': input3,
        'filters': filters3,
        'stride': stride3,
        'padding': padding3,
        'data_format': data_format3,
        'dilations': dilations3,
        'name': name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    input4 = np.random.rand(3, 12, 3).astype(np.float32)
    filters4 = np.random.rand(4, 3, 6).astype(np.float32)
    stride4 = [3]
    padding4 = 'SAME'
    data_format4 = 'NWC'
    dilations4 = [1]
    name4 = 'conv1d_4'

    input_dict4 = {
        'input': input4,
        'filters': filters4,
        'stride': stride4,
        'padding': padding4,
        'data_format': data_format4,
        'dilations': dilations4,
        'name': name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    input5 = np.random.rand(1, 8, 1).astype(np.float32)
    filters5 = np.random.rand(2, 1, 2).astype(np.float32)
    stride5 = [1]
    padding5 = 'VALID'
    data_format5 = 'NWC'
    dilations5 = [3]
    name5 = 'conv1d_5'

    input_dict5 = {
        'input': input5,
        'filters': filters5,
        'stride': stride5,
        'padding': padding5,
        'data_format': data_format5,
        'dilations': dilations5,
        'name': name5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6 - NCW format
    input6 = np.random.rand(1, 3, 10).astype(np.float32)
    filters6 = np.random.rand(3, 3, 5).astype(np.float32)
    stride6 = [1]
    padding6 = 'VALID'
    data_format6 = 'NCW'
    dilations6 = [1]
    name6 = 'conv1d_6'

    input_dict6 = {
        'input': input6,
        'filters': filters6,
        'stride': stride6,
        'padding': padding6,
        'data_format': data_format6,
        'dilations': dilations6,
        'name': name6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7 - Float16
    input7 = np.random.rand(1, 10, 3).astype(np.float16)
    filters7 = np.random.rand(3, 3, 5).astype(np.float16)
    stride7 = [1]
    padding7 = 'VALID'
    data_format7 = 'NWC'
    dilations7 = [1]
    name7 = 'conv1d_7'

    input_dict7 = {
        'input': input7,
        'filters': filters7,
        'stride': stride7,
        'padding': padding7,
        'data_format': data_format7,
        'dilations': dilations7,
        'name': name7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8 - Float64
    input8 = np.random.rand(1, 10, 3).astype(np.float64)
    filters8 = np.random.rand(3, 3, 5).astype(np.float64)
    stride8 = [1]
    padding8 = 'VALID'
    data_format8 = 'NWC'
    dilations8 = [1]
    name8 = 'conv1d_8'

    input_dict8 = {
        'input': input8,
        'filters': filters8,
        'stride': stride8,
        'padding': padding8,
        'data_format': data_format8,
        'dilations': dilations8,
        'name': name8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

     # Input 9
    input9 = np.random.rand(1, 5, 1).astype(np.float32)
    filters9 = np.random.rand(2, 1, 1).astype(np.float32)
    stride9 = [1]
    padding9 = 'SAME'
    data_format9 = 'NWC'
    dilations9 = [1]
    name9 = 'conv1d_9'

    input_dict9 = {
        'input': input9,
        'filters': filters9,
        'stride': stride9,
        'padding': padding9,
        'data_format': data_format9,
        'dilations': dilations9,
        'name': name9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10
    input10 = np.random.rand(2, 10, 2).astype(np.float32)
    filters10 = np.random.rand(3, 2, 4).astype(np.float32)
    stride10 = [2]
    padding10 = 'VALID'
    data_format10 = 'NWC'
    dilations10 = [1]
    name10 = 'conv1d_10'

    input_dict10 = {
        'input': input10,
        'filters': filters10,
        'stride': stride10,
        'padding': padding10,
        'data_format': data_format10,
        'dilations': dilations10,
        'name': name10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.conv1d_2"] = tf_nn_conv1d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.conv1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.conv1d_2'.")

check_valid('tf.nn.conv1d', generated_inputs['tf.nn.conv1d_2'], lib="tf", suffix=2)
