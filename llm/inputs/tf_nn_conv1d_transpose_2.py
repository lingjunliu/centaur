
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
    filters1 = np.random.rand(3, 3, 3).astype(np.float32)
    output_shape1 = np.array([1, 5, 3]).astype(np.int32)
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
    input2 = np.random.rand(2, 10, 2).astype(np.float32)
    filters2 = np.random.rand(5, 2, 2).astype(np.float32)
    output_shape2 = np.array([2, 14, 2]).astype(np.int32)
    strides2 = [2]
    padding2 = 'VALID'
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
