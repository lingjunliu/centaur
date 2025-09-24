
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_erosion2d_inputs():
    list_of_inputs = []

    # Input 1
    value = np.random.rand(1, 10, 10, 3).astype(np.float32)
    filters = np.random.rand(3, 3, 3).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "SAME"
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "erosion1"

    input_dict = {
        "value": value,
        "filters": filters,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    value = np.random.rand(2, 5, 5, 1).astype(np.float32)
    filters = np.random.rand(2, 2, 1).astype(np.float32)
    strides = [1, 2, 2, 1]
    padding = "VALID"
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "erosion2"

    input_dict = {
        "value": value,
        "filters": filters,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    value = np.random.rand(1, 7, 7, 5).astype(np.float32)
    filters = np.random.rand(4, 4, 5).astype(np.float32)
    strides = [1, 1, 2, 1]
    padding = "SAME"
    data_format = "NHWC"
    dilations = [1, 2, 1, 1]
    name = "erosion3"

    input_dict = {
        "value": value,
        "filters": filters,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4
    value = np.random.rand(4, 12, 12, 2).astype(np.float32)
    filters = np.random.rand(5, 5, 2).astype(np.float32)
    strides = [1, 3, 1, 1]
    padding = "VALID"
    data_format = "NHWC"
    dilations = [1, 1, 2, 1]
    name = None

    input_dict = {
        "value": value,
        "filters": filters,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    value = np.random.rand(1, 8, 8, 4).astype(np.float32)
    filters = np.random.rand(2, 2, 4).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "SAME"
    data_format = "NHWC"
    dilations = [1, 3, 3, 1]
    name = "erosion5"

    input_dict = {
        "value": value,
        "filters": filters,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: different filter size
    value = np.random.rand(1, 10, 10, 3).astype(np.float32)
    filters = np.random.rand(1, 1, 3).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "SAME"
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "erosion6"
    input_dict = {
        "value": value,
        "filters": filters,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: different dilation rates
    value = np.random.rand(1, 10, 10, 3).astype(np.float32)
    filters = np.random.rand(3, 3, 3).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "SAME"
    data_format = "NHWC"
    dilations = [1, 2, 3, 1]
    name = "erosion7"
    input_dict = {
        "value": value,
        "filters": filters,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: different strides
    value = np.random.rand(1, 10, 10, 3).astype(np.float32)
    filters = np.random.rand(3, 3, 3).astype(np.float32)
    strides = [1, 2, 3, 1]
    padding = "SAME"
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "erosion8"
    input_dict = {
        "value": value,
        "filters": filters,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: grayscale image
    value = np.random.rand(1, 10, 10, 1).astype(np.float32)
    filters = np.random.rand(3, 3, 1).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "SAME"
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "erosion9"
    input_dict = {
        "value": value,
        "filters": filters,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: small image and filter
    value = np.random.rand(1, 4, 4, 3).astype(np.float32)
    filters = np.random.rand(2, 2, 3).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "VALID"
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "erosion10"
    input_dict = {
        "value": value,
        "filters": filters,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.erosion2d"] = tf_nn_erosion2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.erosion2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.erosion2d'.")

check_valid('tf.nn.erosion2d', generated_inputs['tf.nn.erosion2d'], lib="tf", suffix=0)
