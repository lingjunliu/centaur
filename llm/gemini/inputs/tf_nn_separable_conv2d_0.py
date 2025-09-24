
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_separable_conv2d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 32, 32, 3).astype(np.float32)
    depthwise_filter_tensor = np.random.rand(3, 3, 3, 1).astype(np.float32)
    pointwise_filter_tensor = np.random.rand(1, 1, 3, 64).astype(np.float32)
    strides_list = [1, 1, 1, 1]
    padding_string = "SAME"
    data_format_string = "NHWC"
    dilations_list = [1, 1]
    name_string = "separable_conv1"

    input_dict = {
        "input": input_tensor,
        "depthwise_filter": depthwise_filter_tensor,
        "pointwise_filter": pointwise_filter_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(4, 64, 64, 16).astype(np.float32)
    depthwise_filter_tensor = np.random.rand(5, 5, 16, 2).astype(np.float32)
    pointwise_filter_tensor = np.random.rand(1, 1, 32, 32).astype(np.float32)
    strides_list = [1, 2, 2, 1]
    padding_string = "VALID"
    data_format_string = "NHWC"
    dilations_list = [1, 1]
    name_string = "separable_conv2"

    input_dict = {
        "input": input_tensor,
        "depthwise_filter": depthwise_filter_tensor,
        "pointwise_filter": pointwise_filter_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(2, 128, 128, 8).astype(np.float32)
    depthwise_filter_tensor = np.random.rand(7, 7, 8, 1).astype(np.float32)
    pointwise_filter_tensor = np.random.rand(1, 1, 8, 16).astype(np.float32)
    strides_list = [1, 1, 1, 1]
    padding_string = "SAME"
    data_format_string = "NHWC"
    dilations_list = [2, 2]
    name_string = "separable_conv3"

    input_dict = {
        "input": input_tensor,
        "depthwise_filter": depthwise_filter_tensor,
        "pointwise_filter": pointwise_filter_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(1, 32, 32, 3).astype(np.float32)
    depthwise_filter_tensor = np.random.rand(3, 3, 3, 1).astype(np.float32)
    pointwise_filter_tensor = np.random.rand(1, 1, 3, 64).astype(np.float32)
    strides_list = [1, 1, 1, 1]
    padding_string = "VALID"
    data_format_string = "NHWC"
    dilations_list = [1, 1]
    name_string = "separable_conv4"

    input_dict = {
        "input": input_tensor,
        "depthwise_filter": depthwise_filter_tensor,
        "pointwise_filter": pointwise_filter_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(1, 32, 32, 3).astype(np.float32)
    depthwise_filter_tensor = np.random.rand(3, 3, 3, 1).astype(np.float32)
    pointwise_filter_tensor = np.random.rand(1, 1, 3, 64).astype(np.float32)
    strides_list = [1, 2, 2, 1]
    padding_string = "SAME"
    data_format_string = "NHWC"
    dilations_list = [1, 1]
    name_string = "separable_conv5"

    input_dict = {
        "input": input_tensor,
        "depthwise_filter": depthwise_filter_tensor,
        "pointwise_filter": pointwise_filter_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    input_tensor = np.random.rand(2, 64, 64, 32).astype(np.float32)
    depthwise_filter_tensor = np.random.rand(3, 3, 32, 1).astype(np.float32)
    pointwise_filter_tensor = np.random.rand(1, 1, 32, 16).astype(np.float32)
    strides_list = [1, 1, 1, 1]
    padding_string = "SAME"
    data_format_string = "NHWC"
    dilations_list = [1, 1]
    name_string = "separable_conv6"

    input_dict = {
        "input": input_tensor,
        "depthwise_filter": depthwise_filter_tensor,
        "pointwise_filter": pointwise_filter_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    input_tensor = np.random.rand(1, 128, 128, 1).astype(np.float32)
    depthwise_filter_tensor = np.random.rand(5, 5, 1, 1).astype(np.float32)
    pointwise_filter_tensor = np.random.rand(1, 1, 1, 32).astype(np.float32)
    strides_list = [1, 2, 2, 1]
    padding_string = "VALID"
    data_format_string = "NHWC"
    dilations_list = [1, 1]
    name_string = "separable_conv7"

    input_dict = {
        "input": input_tensor,
        "depthwise_filter": depthwise_filter_tensor,
        "pointwise_filter": pointwise_filter_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.random.rand(1, 32, 32, 3).astype(np.float32)
    depthwise_filter_tensor = np.random.rand(3, 3, 3, 1).astype(np.float32)
    pointwise_filter_tensor = np.random.rand(1, 1, 3, 64).astype(np.float32)
    strides_list = [1, 1, 1, 1]
    padding_string = "SAME"
    data_format_string = "NHWC" # keep NHWC because explicit padding is used
    dilations_list = [1, 1]
    name_string = "separable_conv8"

    input_dict = {
        "input": input_tensor,
        "depthwise_filter": depthwise_filter_tensor,
        "pointwise_filter": pointwise_filter_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(1, 32, 32, 3).astype(np.float32)
    depthwise_filter_tensor = np.random.rand(3, 3, 3, 1).astype(np.float32)
    pointwise_filter_tensor = np.random.rand(1, 1, 3, 64).astype(np.float32)
    strides_list = [1, 1, 1, 1]
    padding_string = "VALID"
    data_format_string = "NHWC" # keep NHWC because explicit padding is used
    dilations_list = [1, 1]
    name_string = "separable_conv9"

    input_dict = {
        "input": input_tensor,
        "depthwise_filter": depthwise_filter_tensor,
        "pointwise_filter": pointwise_filter_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.random.rand(1, 32, 32, 3).astype(np.float32)
    depthwise_filter_tensor = np.random.rand(3, 3, 3, 1).astype(np.float32)
    pointwise_filter_tensor = np.random.rand(1, 1, 3, 64).astype(np.float32)
    strides_list = [1, 2, 2, 1]
    padding_string = "VALID"
    data_format_string = "NHWC" # keep NHWC because explicit padding is used
    dilations_list = [1, 1]
    name_string = "separable_conv10"

    input_dict = {
        "input": input_tensor,
        "depthwise_filter": depthwise_filter_tensor,
        "pointwise_filter": pointwise_filter_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_tensor = np.random.rand(1, 64, 64, 4).astype(np.float32)
    depthwise_filter_tensor = np.random.rand(3, 3, 4, 1).astype(np.float32)
    pointwise_filter_tensor = np.random.rand(1, 1, 4, 8).astype(np.float32)
    strides_list = [1, 1, 1, 1]
    padding_string = "SAME"
    data_format_string = "NHWC"
    dilations_list = [1, 1]
    name_string = "separable_conv11"

    input_dict = {
        "input": input_tensor,
        "depthwise_filter": depthwise_filter_tensor,
        "pointwise_filter": pointwise_filter_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Explicit padding
    input_tensor = np.random.rand(1, 32, 32, 3).astype(np.float32)
    depthwise_filter_tensor = np.random.rand(3, 3, 3, 1).astype(np.float32)
    pointwise_filter_tensor = np.random.rand(1, 1, 3, 64).astype(np.float32)
    strides_list = [1, 1, 1, 1]
    padding_string = [[0, 0], [1, 1], [1, 1], [0, 0]]
    data_format_string = "NHWC"
    dilations_list = [1, 1]
    name_string = "separable_conv12"

    input_dict = {
        "input": input_tensor,
        "depthwise_filter": depthwise_filter_tensor,
        "pointwise_filter": pointwise_filter_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.separable_conv2d"] = tf_nn_separable_conv2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.separable_conv2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.separable_conv2d'.")

check_valid('tf.nn.separable_conv2d', generated_inputs['tf.nn.separable_conv2d'], lib="tf", suffix=0)
