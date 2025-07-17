
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_quantized_avg_pool_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.int8)
    min_input_tensor = np.array([0.0], dtype=np.float32)
    max_input_tensor = np.array([5.0], dtype=np.float32)
    ksize = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    padding = "VALID"

    input_dict = {
        "name": "avg_pool_1",
        "input": input_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.uint8)
    min_input_tensor = np.array([0.0], dtype=np.float32)
    max_input_tensor = np.array([255.0], dtype=np.float32)
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = "SAME"

    input_dict = {
        "name": "avg_pool_2",
        "input": input_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.int32)
    min_input_tensor = np.array([-10.0], dtype=np.float32)
    max_input_tensor = np.array([10.0], dtype=np.float32)
    ksize = [1, 1, 1, 1]
    strides = [1, 1, 1, 1]
    padding = "VALID"

    input_dict = {
        "name": "avg_pool_3",
        "input": input_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 4
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int16)
    min_input_tensor = np.array([-5.0], dtype=np.float32)
    max_input_tensor = np.array([15.0], dtype=np.float32)
    ksize = [1, 1, 1, 1]
    strides = [1, 1, 1, 1]
    padding = "SAME"

    input_dict = {
        "name": "avg_pool_4",
        "input": input_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.uint16)
    min_input_tensor = np.array([0.0], dtype=np.float32)
    max_input_tensor = np.array([65535.0], dtype=np.float32)
    ksize = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    padding = "VALID"

    input_dict = {
        "name": "avg_pool_5",
        "input": input_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.int8)
    min_input_tensor = np.array([-128.0], dtype=np.float32)
    max_input_tensor = np.array([127.0], dtype=np.float32)
    ksize = [1, 2, 1, 1]
    strides = [1, 1, 1, 1]
    padding = "SAME"

    input_dict = {
        "name": "avg_pool_6",
        "input": input_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]], dtype=np.uint8)
    min_input_tensor = np.array([0.0], dtype=np.float32)
    max_input_tensor = np.array([200.0], dtype=np.float32)
    ksize = [2, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = "VALID"

    input_dict = {
        "name": "avg_pool_7",
        "input": input_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]], dtype=np.int32)
    min_input_tensor = np.array([-20.0], dtype=np.float32)
    max_input_tensor = np.array([30.0], dtype=np.float32)
    ksize = [2, 1, 2, 1]
    strides = [1, 1, 1, 1]
    padding = "SAME"

    input_dict = {
        "name": "avg_pool_8",
        "input": input_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]], dtype=np.int16)
    min_input_tensor = np.array([-10.0], dtype=np.float32)
    max_input_tensor = np.array([10.0], dtype=np.float32)
    ksize = [2, 2, 1, 1]
    strides = [1, 1, 1, 1]
    padding = "VALID"

    input_dict = {
        "name": "avg_pool_9",
        "input": input_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]], dtype=np.uint16)
    min_input_tensor = np.array([0.0], dtype=np.float32)
    max_input_tensor = np.array([50000.0], dtype=np.float32)
    ksize = [2, 1, 1, 1]
    strides = [1, 1, 1, 1]
    padding = "SAME"

    input_dict = {
        "name": "avg_pool_10",
        "input": input_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedAvgPool"] = tf_raw_ops_quantized_avg_pool_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizedAvgPool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedAvgPool'.")

check_valid('tf.raw_ops.QuantizedAvgPool', generated_inputs['tf.raw_ops.QuantizedAvgPool'], lib="tf", suffix=0)
