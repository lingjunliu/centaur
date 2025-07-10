
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_avg_pool2d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 4, 4, 1).astype(np.float32)
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'VALID'
    data_format = 'NHWC'
    name = None

    input_dict = {
        "input": tf.convert_to_tensor(input_tensor),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(1, 8, 8, 3).astype(np.float32)
    ksize = [1, 4, 4, 1]
    strides = [1, 2, 2, 1]
    padding = 'SAME'
    data_format = 'NHWC'
    name = "avg_pool_1"

    input_dict = {
        "input": tf.convert_to_tensor(input_tensor),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(2, 16, 16, 5).astype(np.float32)
    ksize = [1, 8, 8, 1]
    strides = [1, 4, 4, 1]
    padding = 'VALID'
    data_format = 'NHWC'
    name = None

    input_dict = {
        "input": tf.convert_to_tensor(input_tensor),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(1, 32, 32, 7).astype(np.float32)
    ksize = [1, 16, 16, 1]
    strides = [1, 8, 8, 1]
    padding = 'SAME'
    data_format = 'NHWC'
    name = "avg_pool_2"

    input_dict = {
        "input": tf.convert_to_tensor(input_tensor),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    input_tensor = np.random.rand(1, 4, 4, 1).astype(np.float32)
    ksize = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    padding = 'VALID'
    data_format = 'NHWC'
    name = None

    input_dict = {
        "input": tf.convert_to_tensor(input_tensor),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.rand(1, 8, 8, 3).astype(np.float32)
    ksize = [1, 4, 4, 1]
    strides = [1, 1, 1, 1]
    padding = 'SAME'
    data_format = 'NHWC'
    name = "avg_pool_1"

    input_dict = {
        "input": tf.convert_to_tensor(input_tensor),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.random.rand(2, 16, 16, 5).astype(np.float32)
    ksize = [1, 8, 8, 1]
    strides = [1, 2, 2, 1]
    padding = 'VALID'
    data_format = 'NHWC'
    name = None

    input_dict = {
        "input": tf.convert_to_tensor(input_tensor),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.random.rand(1, 32, 32, 7).astype(np.float32)
    ksize = [1, 16, 16, 1]
    strides = [1, 4, 4, 1]
    padding = 'SAME'
    data_format = 'NHWC'
    name = "avg_pool_2"

    input_dict = {
        "input": tf.convert_to_tensor(input_tensor),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(1, 4, 4, 1).astype(np.float32)
    ksize = [2, 2]
    strides = [2, 2]
    padding = 'VALID'
    data_format = 'NHWC'
    name = None

    input_dict = {
        "input": tf.convert_to_tensor(input_tensor),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.random.rand(1, 8, 8, 3).astype(np.float32)
    ksize = [4, 4]
    strides = [2, 2]
    padding = 'SAME'
    data_format = 'NHWC'
    name = "avg_pool_1"

    input_dict = {
        "input": tf.convert_to_tensor(input_tensor),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    input_tensor = np.random.rand(1, 8, 8, 3).astype(np.float32)
    ksize = [1]
    strides = [1]
    padding = 'SAME'
    data_format = 'NHWC'
    name = "avg_pool_1"

    input_dict = {
        "input": tf.convert_to_tensor(input_tensor),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.avg_pool2d"] = tf_nn_avg_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.avg_pool2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.avg_pool2d'.")

check_valid('tf.nn.avg_pool2d', generated_inputs['tf.nn.avg_pool2d'], lib="tf", suffix=0)
