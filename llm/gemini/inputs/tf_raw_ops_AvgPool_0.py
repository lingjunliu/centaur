
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_avgpool_inputs():
    list_of_inputs = []

    # Input 1
    value = np.random.rand(1, 5, 5, 3).astype(np.float32)
    ksize = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    padding = "VALID"
    data_format = "NHWC"
    name = "avg_pool_1"

    input_dict = {
        "value": value,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    value = np.random.rand(1, 10, 10, 1).astype(np.float32)
    ksize = [1, 3, 3, 1]
    strides = [1, 2, 2, 1]
    padding = "SAME"
    data_format = "NHWC"
    name = "avg_pool_2"

    input_dict = {
        "value": value,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    value = np.random.rand(2, 8, 8, 3).astype(np.float32)
    ksize = [1, 4, 4, 1]
    strides = [1, 3, 3, 1]
    padding = "VALID"
    data_format = "NHWC"
    name = "avg_pool_3"

    input_dict = {
        "value": value,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    value = np.random.rand(2, 6, 6, 5).astype(np.float32)
    ksize = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    padding = "SAME"
    data_format = "NHWC"
    name = "avg_pool_4"

    input_dict = {
        "value": value,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 (NCHW)
    value = np.random.rand(1, 3, 5, 5).astype(np.float32)
    ksize = [1, 1, 2, 2]
    strides = [1, 1, 1, 1]
    padding = "VALID"
    data_format = "NCHW"
    name = "avg_pool_5"

    input_dict = {
        "value": value,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (float16)
    value = np.random.rand(1, 5, 5, 3).astype(np.float16)
    ksize = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    padding = "VALID"
    data_format = "NHWC"
    name = "avg_pool_6"

    input_dict = {
        "value": value,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (half)
    value = np.random.rand(1, 5, 5, 3).astype(np.float16)
    ksize = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    padding = "VALID"
    data_format = "NHWC"
    name = "avg_pool_7"

    input_dict = {
        "value": value,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (float64)
    value = np.random.rand(1, 5, 5, 3).astype(np.float64)
    ksize = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    padding = "VALID"
    data_format = "NHWC"
    name = "avg_pool_8"

    input_dict = {
        "value": value,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    value = np.random.rand(1, 7, 7, 3).astype(np.float32)
    ksize = [1, 7, 7, 1]
    strides = [1, 1, 1, 1]
    padding = "VALID"
    data_format = "NHWC"
    name = "avg_pool_9"

    input_dict = {
        "value": value,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    value = np.random.rand(1, 3, 3, 1).astype(np.float32)
    ksize = [1, 1, 1, 1]
    strides = [1, 1, 1, 1]
    padding = "SAME"
    data_format = "NHWC"
    name = "avg_pool_10"

    input_dict = {
        "value": value,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AvgPool"] = tf_raw_ops_avgpool_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.AvgPool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AvgPool'.")

check_valid('tf.raw_ops.AvgPool', generated_inputs['tf.raw_ops.AvgPool'], lib="tf", suffix=0)
