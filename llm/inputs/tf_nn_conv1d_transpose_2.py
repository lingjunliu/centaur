
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_conv1d_transpose_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.random.rand(1, 5, 3).astype(np.float32)
    filters1 = np.random.rand(2, 4, 3).astype(np.float32)
    output_shape1 = np.array([1, 6, 4]).astype(np.int32)
    strides1 = [1]
    padding1 = 'SAME'
    data_format1 = 'NWC'
    dilations1 = [1]
    name1 = 'conv1d_transpose_1'

    input_dict1 = {
        "input": input1,
        "filters": filters1,
        "output_shape": output_shape1,
        "strides": strides1,
        "padding": padding1,
        "data_format": data_format1,
        "dilations": dilations1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.random.rand(1, 3, 2).astype(np.float32)
    filters2 = np.random.rand(2, 3, 2).astype(np.float32)
    output_shape2 = np.array([1, 4, 3]).astype(np.int32)
    strides2 = [1]
    padding2 = 'SAME'
    data_format2 = 'NWC'
    dilations2 = [1]
    name2 = 'conv1d_transpose_2'

    input_dict2 = {
        "input": input2,
        "filters": filters2,
        "output_shape": output_shape2,
        "strides": strides2,
        "padding": padding2,
        "data_format": data_format2,
        "dilations": dilations2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3 : strides = 2 and valid padding to avoid descriptor creation issues
    input3 = np.random.rand(1, 5, 3).astype(np.float32)
    filters3 = np.random.rand(3, 2, 3).astype(np.float32)
    output_shape3 = np.array([1, 9, 2]).astype(np.int32)
    strides3 = [2]
    padding3 = 'VALID'
    data_format3 = 'NWC'
    dilations3 = [1]
    name3 = 'conv1d_transpose_3'

    input_dict3 = {
        "input": input3,
        "filters": filters3,
        "output_shape": output_shape3,
        "strides": strides3,
        "padding": padding3,
        "data_format": data_format3,
        "dilations": dilations3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4 : strides = 1 and valid padding
    input4 = np.random.rand(1, 5, 3).astype(np.float32)
    filters4 = np.random.rand(3, 2, 3).astype(np.float32)
    output_shape4 = np.array([1, 7, 2]).astype(np.int32)
    strides4 = [1]
    padding4 = 'VALID'
    data_format4 = 'NWC'
    dilations4 = [1]
    name4 = 'conv1d_transpose_4'

    input_dict4 = {
        "input": input4,
        "filters": filters4,
        "output_shape": output_shape4,
        "strides": strides4,
        "padding": padding4,
        "data_format": data_format4,
        "dilations": dilations4,
        "name": name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.conv1d_transpose_2"] = tf_nn_conv1d_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.conv1d_transpose_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.conv1d_transpose_2'.")

check_valid('tf.nn.conv1d_transpose', generated_inputs['tf.nn.conv1d_transpose_2'], lib="tf", suffix=2)
