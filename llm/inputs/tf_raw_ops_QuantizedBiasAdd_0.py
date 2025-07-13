
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_quantized_bias_add_inputs():
    list_of_inputs = []

    # Input 1, valid
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
    bias_tensor = np.array([1, 2, 3], dtype=np.int8)
    min_input_tensor = np.array([-1.0], dtype=np.float32)
    max_input_tensor = np.array([6.0], dtype=np.float32)
    min_bias_tensor = np.array([-1.0], dtype=np.float32)
    max_bias_tensor = np.array([3.0], dtype=np.float32)
    out_type = tf.qint8

    input_dict = {
        "input": input_tensor,
        "bias": bias_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "min_bias": min_bias_tensor,
        "max_bias": max_bias_tensor,
        "out_type": out_type,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)
    bias_tensor = np.array([1, 2], dtype=np.uint8)
    min_input_tensor = np.array([0.0], dtype=np.float32)
    max_input_tensor = np.array([255.0], dtype=np.float32)
    min_bias_tensor = np.array([0.0], dtype=np.float32)
    max_bias_tensor = np.array([2.0], dtype=np.float32)
    out_type = tf.quint8

    input_dict = {
        "input": input_tensor,
        "bias": bias_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "min_bias": min_bias_tensor,
        "max_bias": max_bias_tensor,
        "out_type": out_type,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    input_tensor = np.array([[1, 2, 3, 4]], dtype=np.int32)
    bias_tensor = np.array([1, 2, 3, 4], dtype=np.int32)
    min_input_tensor = np.array([-2147483648.0], dtype=np.float32)
    max_input_tensor = np.array([2147483647.0], dtype=np.float32)
    min_bias_tensor = np.array([-100.0], dtype=np.float32)
    max_bias_tensor = np.array([100.0], dtype=np.float32)
    out_type = tf.qint32

    input_dict = {
        "input": input_tensor,
        "bias": bias_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "min_bias": min_bias_tensor,
        "max_bias": max_bias_tensor,
        "out_type": out_type,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    input_tensor = np.array([1, 2, 3, 4, 5], dtype=np.int16)
    bias_tensor = np.array([1, 2, 3, 4, 5], dtype=np.int16)
    min_input_tensor = np.array([-32768.0], dtype=np.float32)
    max_input_tensor = np.array([32767.0], dtype=np.float32)
    min_bias_tensor = np.array([-10.0], dtype=np.float32)
    max_bias_tensor = np.array([10.0], dtype=np.float32)
    out_type = tf.qint16

    input_dict = {
        "input": input_tensor,
        "bias": bias_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "min_bias": min_bias_tensor,
        "max_bias": max_bias_tensor,
        "out_type": out_type,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    bias_tensor = np.array([1, 2], dtype=np.uint16)
    min_input_tensor = np.array([0.0], dtype=np.float32)
    max_input_tensor = np.array([65535.0], dtype=np.float32)
    min_bias_tensor = np.array([0.0], dtype=np.float32)
    max_bias_tensor = np.array([5.0], dtype=np.float32)
    out_type = tf.quint16

    input_dict = {
        "input": input_tensor,
        "bias": bias_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "min_bias": min_bias_tensor,
        "max_bias": max_bias_tensor,
        "out_type": out_type,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid, different shape
    input_tensor = np.array([[[1, 2, 3, 4, 5]]], dtype=np.int8)
    bias_tensor = np.array([1, 2, 3, 4, 5], dtype=np.int8)
    min_input_tensor = np.array([-128.0], dtype=np.float32)
    max_input_tensor = np.array([127.0], dtype=np.float32)
    min_bias_tensor = np.array([-5.0], dtype=np.float32)
    max_bias_tensor = np.array([5.0], dtype=np.float32)
    out_type = tf.qint8

    input_dict = {
        "input": input_tensor,
        "bias": bias_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "min_bias": min_bias_tensor,
        "max_bias": max_bias_tensor,
        "out_type": out_type,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid, different ranges
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    bias_tensor = np.array([1, 2, 3], dtype=np.uint8)
    min_input_tensor = np.array([50.0], dtype=np.float32)
    max_input_tensor = np.array([100.0], dtype=np.float32)
    min_bias_tensor = np.array([1.0], dtype=np.float32)
    max_bias_tensor = np.array([5.0], dtype=np.float32)
    out_type = tf.quint8

    input_dict = {
        "input": input_tensor,
        "bias": bias_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "min_bias": min_bias_tensor,
        "max_bias": max_bias_tensor,
        "out_type": out_type,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8, valid
    input_tensor = np.array([[1000, 2000]], dtype=np.int32)
    bias_tensor = np.array([1000, 2000], dtype=np.int32)
    min_input_tensor = np.array([-10000.0], dtype=np.float32)
    max_input_tensor = np.array([10000.0], dtype=np.float32)
    min_bias_tensor = np.array([-500.0], dtype=np.float32)
    max_bias_tensor = np.array([500.0], dtype=np.float32)
    out_type = tf.qint32

    input_dict = {
        "input": input_tensor,
        "bias": bias_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "min_bias": min_bias_tensor,
        "max_bias": max_bias_tensor,
        "out_type": out_type,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid
    input_tensor = np.array([[-5, -4, -3], [-2, -1, 0]], dtype=np.int16)
    bias_tensor = np.array([-1, 0, 1], dtype=np.int16)
    min_input_tensor = np.array([-10.0], dtype=np.float32)
    max_input_tensor = np.array([5.0], dtype=np.float32)
    min_bias_tensor = np.array([-2.0], dtype=np.float32)
    max_bias_tensor = np.array([2.0], dtype=np.float32)
    out_type = tf.qint16

    input_dict = {
        "input": input_tensor,
        "bias": bias_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "min_bias": min_bias_tensor,
        "max_bias": max_bias_tensor,
        "out_type": out_type,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid
    input_tensor = np.array([[[10, 20], [30, 40]]], dtype=np.uint16)
    bias_tensor = np.array([10, 20], dtype=np.uint16)
    min_input_tensor = np.array([0.0], dtype=np.float32)
    max_input_tensor = np.array([100.0], dtype=np.float32)
    min_bias_tensor = np.array([0.0], dtype=np.float32)
    max_bias_tensor = np.array([10.0], dtype=np.float32)
    out_type = tf.quint16

    input_dict = {
        "input": input_tensor,
        "bias": bias_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "min_bias": min_bias_tensor,
        "max_bias": max_bias_tensor,
        "out_type": out_type,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11, valid, 1D input
    input_tensor = np.array([1, 2, 3, 4], dtype=np.int8)
    bias_tensor = np.array([1, 2, 3, 4], dtype=np.int8)
    min_input_tensor = np.array([-1.0], dtype=np.float32)
    max_input_tensor = np.array([6.0], dtype=np.float32)
    min_bias_tensor = np.array([-1.0], dtype=np.float32)
    max_bias_tensor = np.array([3.0], dtype=np.float32)
    out_type = tf.qint8

    input_dict = {
        "input": input_tensor,
        "bias": bias_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "min_bias": min_bias_tensor,
        "max_bias": max_bias_tensor,
        "out_type": out_type,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_quantized_bias_add_inputs()
for i, input_dict in enumerate(inputs):
    generated_inputs.setdefault("tf.raw_ops.QuantizedBiasAdd", []).append(input_dict)

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizedBiasAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedBiasAdd'.")

check_valid('tf.raw_ops.QuantizedBiasAdd', generated_inputs['tf.raw_ops.QuantizedBiasAdd'], lib="tf", suffix=0)
