
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_avg_pool2d_inputs():
    list_of_inputs = []

    # Input 1
    input_1 = np.random.randn(1, 4, 4, 1).astype(np.float32)
    ksize_1 = [2, 2]
    strides_1 = [2, 2]
    padding_1 = "VALID"
    data_format_1 = "NHWC"
    name_1 = "avg_pool_1"
    list_of_inputs.append({
        "input": input_1,
        "ksize": ksize_1,
        "strides": strides_1,
        "padding": padding_1,
        "data_format": data_format_1,
        "name": name_1
    })

    # Input 2
    input_2 = np.random.randn(2, 8, 8, 3).astype(np.float32)
    ksize_2 = [1, 2, 2, 1]
    strides_2 = [1, 2, 2, 1]
    padding_2 = "SAME"
    data_format_2 = "NHWC"
    name_2 = "avg_pool_2"
    list_of_inputs.append({
        "input": input_2,
        "ksize": ksize_2,
        "strides": strides_2,
        "padding": padding_2,
        "data_format": data_format_2,
        "name": name_2
    })

    # Input 3
    input_3 = np.random.randn(1, 10, 10, 2).astype(np.float64)
    ksize_3 = [1, 3, 3, 1]
    strides_3 = [1, 2, 2, 1]
    padding_3 = "VALID"
    data_format_3 = "NHWC"
    name_3 = "avg_pool_3"
    list_of_inputs.append({
        "input": input_3,
        "ksize": ksize_3,
        "strides": strides_3,
        "padding": padding_3,
        "data_format": data_format_3,
        "name": name_3
    })

    # Input 4
    input_4 = np.random.randn(4, 16, 16, 64).astype(np.float32)
    ksize_4 = [4, 4]
    strides_4 = [4, 4]
    padding_4 = "SAME"
    data_format_4 = "NHWC"
    name_4 = "avg_pool_4"
    list_of_inputs.append({
        "input": input_4,
        "ksize": ksize_4,
        "strides": strides_4,
        "padding": padding_4,
        "data_format": data_format_4,
        "name": name_4
    })

    # Input 5
    input_5 = np.random.randn(1, 5, 5, 3).astype(np.float32)
    ksize_5 = [2, 2]
    strides_5 = [1, 1]
    padding_5 = "SAME"
    data_format_5 = "NHWC"
    name_5 = "avg_pool_5"
    list_of_inputs.append({
        "input": input_5,
        "ksize": ksize_5,
        "strides": strides_5,
        "padding": padding_5,
        "data_format": data_format_5,
        "name": name_5
    })

    # Input 6
    input_6 = np.random.randn(2, 14, 14, 3).astype(np.float32)
    ksize_6 = [2, 2]
    strides_6 = [2, 2]
    padding_6 = "VALID"
    data_format_6 = "NHWC"
    name_6 = "avg_pool_6"
    list_of_inputs.append({
        "input": input_6,
        "ksize": ksize_6,
        "strides": strides_6,
        "padding": padding_6,
        "data_format": data_format_6,
        "name": name_6
    })

    # Input 7
    input_7 = np.random.randn(1, 1, 1, 1).astype(np.float32)
    ksize_7 = [1, 1]
    strides_7 = [1, 1]
    padding_7 = "VALID"
    data_format_7 = "NHWC"
    name_7 = "avg_pool_7"
    list_of_inputs.append({
        "input": input_7,
        "ksize": ksize_7,
        "strides": strides_7,
        "padding": padding_7,
        "data_format": data_format_7,
        "name": name_7
    })

    # Input 8
    input_8 = np.random.randn(8, 32, 32, 3).astype(np.float32)
    ksize_8 = [2, 2]
    strides_8 = [1, 1]
    padding_8 = "SAME"
    data_format_8 = "NHWC"
    name_8 = "avg_pool_8"
    list_of_inputs.append({
        "input": input_8,
        "ksize": ksize_8,
        "strides": strides_8,
        "padding": padding_8,
        "data_format": data_format_8,
        "name": name_8
    })

    # Input 9
    input_9 = np.random.randn(2, 14, 14, 128).astype(np.float64)
    ksize_9 = [1, 3, 3, 1]
    strides_9 = [1, 1, 1, 1]
    padding_9 = "VALID"
    data_format_9 = "NHWC"
    name_9 = "avg_pool_9"
    list_of_inputs.append({
        "input": input_9,
        "ksize": ksize_9,
        "strides": strides_9,
        "padding": padding_9,
        "data_format": data_format_9,
        "name": name_9
    })

    # Input 10
    input_10 = np.random.randn(3, 28, 28, 16).astype(np.float32)
    ksize_10 = [1, 2, 2, 1]
    strides_10 = [1, 2, 2, 1]
    padding_10 = "SAME"
    data_format_10 = "NHWC"
    name_10 = "avg_pool_10"
    list_of_inputs.append({
        "input": input_10,
        "ksize": ksize_10,
        "strides": strides_10,
        "padding": padding_10,
        "data_format": data_format_10,
        "name": name_10
    })

    return list_of_inputs

generated_inputs["tf.nn.avg_pool2d"] = tf_nn_avg_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.avg_pool2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.avg_pool2d'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.avg_pool2d', generated_inputs['tf.nn.avg_pool2d'], lib="tf", suffix=0)
