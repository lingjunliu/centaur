
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool2d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.float32)
    ksize = 2
    strides = [1, 1, 1, 1]
    padding = [[0, 0], [0, 0], [0, 0], [0, 0]]
    data_format = 'NHWC'
    name = "pool1"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=np.float32)
    ksize = 2
    strides = [1, 2, 2, 1]
    padding = "VALID"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": 'NHWC',
        "name": "pool2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.float32)
    ksize = 2
    strides = [2, 2]
    padding = "SAME"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": 'NHWC',
        "name": "pool3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.float32)
    ksize = 1
    strides = [1, 1]
    padding = "VALID"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": 'NHWC',
        "name": "pool4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]]], dtype=np.float32)
    ksize = 2
    strides = [2, 2, 2, 2]
    padding = "VALID"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": 'NHWC',
        "name": "pool5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]]], dtype=np.float32)
    ksize = 2
    strides = [2, 2, 2, 2]
    padding = "SAME"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": 'NHWC',
        "name": "pool6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]]], dtype=np.float32)
    ksize = 2
    strides = [1, 1, 1, 1]
    padding = [[0, 0], [1, 1], [1, 1], [0, 0]]

    input_dict = {
        "input": input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": 'NHWC',
        "name": "pool7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.float32)
    ksize = 2
    strides = [1, 1, 1, 1]
    padding = "VALID"
    data_format = 'NHWC'
    name = "pool8"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.float32)
    ksize = 2
    strides = [1, 1, 1, 1]
    padding = "SAME"
    data_format = 'NHWC'
    name = "pool9"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10
    input_tensor = np.array([[[[1, 2, 3], [4, 5, 6]]]], dtype=np.float32)
    ksize = 1
    strides = [1,1,1,1]
    padding = "VALID"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": 'NHWC',
        "name": "pool10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.max_pool2d_7"] = tf_nn_max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.max_pool2d_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.max_pool2d_7'.")

check_valid('tf.nn.max_pool2d', generated_inputs['tf.nn.max_pool2d_7'], lib="tf", suffix=7)
