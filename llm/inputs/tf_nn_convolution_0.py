
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_convolution_inputs():
    list_of_inputs = []

    # Input 1: Simple 2D convolution
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 3, 1).astype(np.float32)
    strides_val = [1, 1]
    padding_val = "VALID"
    data_format_val = "NHWC"
    dilations_val = [1, 1]
    name_val = "conv1"

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "strides": strides_val,
        "padding": padding_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D convolution with SAME padding
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 3, 1).astype(np.float32)
    strides_val = [1, 1]
    padding_val = "SAME"
    data_format_val = "NHWC"
    dilations_val = [1, 1]
    name_val = "conv2"

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "strides": strides_val,
        "padding": padding_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D convolution with strides > 1
    input_tensor = np.random.rand(1, 7, 7, 3).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 3, 1).astype(np.float32)
    strides_val = [2, 2]
    padding_val = "VALID"
    data_format_val = "NHWC"
    dilations_val = [1, 1]
    name_val = "conv3"

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "strides": strides_val,
        "padding": padding_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D convolution with NCHW data format
    input_tensor = np.random.rand(1, 3, 5, 5).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 3, 1).astype(np.float32)
    strides_val = [1, 1]
    padding_val = "VALID"
    data_format_val = "NCHW"
    dilations_val = [1, 1]
    name_val = "conv4"

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "strides": strides_val,
        "padding": padding_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D convolution with dilation
    input_tensor = np.random.rand(1, 7, 7, 3).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 3, 1).astype(np.float32)
    strides_val = [1, 1]
    padding_val = "VALID"
    data_format_val = "NHWC"
    dilations_val = [2, 2]
    name_val = "conv5"

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "strides": strides_val,
        "padding": padding_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6: 1D convolution
    input_tensor = np.random.rand(1, 10, 3).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 1).astype(np.float32)
    strides_val = [1]
    padding_val = "VALID"
    data_format_val = "NWC"
    dilations_val = [1]
    name_val = "conv6"

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "strides": strides_val,
        "padding": padding_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D convolution
    input_tensor = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 3, 3, 1).astype(np.float32)
    strides_val = [1, 1, 1]
    padding_val = "VALID"
    data_format_val = "NDHWC"
    dilations_val = [1, 1, 1]
    name_val = "conv7"

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "strides": strides_val,
        "padding": padding_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Batch size > 1
    input_tensor = np.random.rand(2, 5, 5, 3).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 3, 1).astype(np.float32)
    strides_val = [1, 1]
    padding_val = "VALID"
    data_format_val = "NHWC"
    dilations_val = [1, 1]
    name_val = "conv8"

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "strides": strides_val,
        "padding": padding_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different filter and input channel sizes
    input_tensor = np.random.rand(1, 5, 5, 5).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 5, 2).astype(np.float32)
    strides_val = [1, 1]
    padding_val = "VALID"
    data_format_val = "NHWC"
    dilations_val = [1, 1]
    name_val = "conv9"

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "strides": strides_val,
        "padding": padding_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: SAME padding with strides > 1
    input_tensor = np.random.rand(1, 7, 7, 3).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 3, 1).astype(np.float32)
    strides_val = [2, 2]
    padding_val = "SAME"
    data_format_val = "NHWC"
    dilations_val = [1, 1]
    name_val = "conv10"

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "strides": strides_val,
        "padding": padding_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.convolution"] = tf_nn_convolution_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.convolution' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.convolution'.")

check_valid('tf.nn.convolution', generated_inputs['tf.nn.convolution'], lib="tf", suffix=0)
