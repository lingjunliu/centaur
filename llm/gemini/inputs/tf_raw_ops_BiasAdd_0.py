
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_bias_add_inputs():
    list_of_inputs = []

    # Input 1, valid
    value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    bias = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    data_format = "NHWC"
    name = "bias_add_1"

    input_dict = {
        "value": value,
        "bias": bias,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    value = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    bias = np.array([0, 1], dtype=np.int32)
    data_format = "NHWC"
    name = "bias_add_2"

    input_dict = {
        "value": value,
        "bias": bias,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid, NCHW
    value = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    bias = np.array([0, 1], dtype=np.int32)
    data_format = "NCHW"
    name = "bias_add_4"

    input_dict = {
        "value": value,
        "bias": bias,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid, negative values
    value = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32)
    bias = np.array([-0.1, -0.2, -0.3], dtype=np.float32)
    data_format = "NHWC"
    name = "bias_add_5"

    input_dict = {
        "value": value,
        "bias": bias,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid, uint8
    value = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    bias = np.array([1, 2, 3], dtype=np.uint8)
    data_format = "NHWC"
    name = "bias_add_6"

    input_dict = {
        "value": value,
        "bias": bias,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid, complex64
    value = np.array([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]], dtype=np.complex64)
    bias = np.array([0.1+0.1j, 0.2+0.2j, 0.3+0.3j], dtype=np.complex64)
    data_format = "NHWC"
    name = "bias_add_7"

    input_dict = {
        "value": value,
        "bias": bias,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid, half
    value = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float16)
    bias = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    data_format = "NHWC"
    name = "bias_add_9"

    input_dict = {
        "value": value,
        "bias": bias,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid, rank 5
    value = np.random.rand(2, 3, 4, 5, 6).astype(np.float32)
    bias = np.random.rand(6).astype(np.float32)
    data_format = "NHWC"
    name = "bias_add_11"

    input_dict = {
        "value": value,
        "bias": bias,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid, rank 2, NCHW
    value = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    bias = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    data_format = "NCHW"
    name = "bias_add_12"

    input_dict = {
        "value": value,
        "bias": bias,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid, int16
    value = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
    bias = np.array([1, 2, 3], dtype=np.int16)
    data_format = "NHWC"
    name = "bias_add_13"

    input_dict = {
        "value": value,
        "bias": bias,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.BiasAdd"] = tf_raw_ops_bias_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.BiasAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BiasAdd'.")

check_valid('tf.raw_ops.BiasAdd', generated_inputs['tf.raw_ops.BiasAdd'], lib="tf", suffix=0)
