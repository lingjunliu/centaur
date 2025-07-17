
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_quantized_concat_inputs():
    list_of_inputs = []

    # Input 1
    concat_dim = np.array(0, dtype=np.int32)
    values = [np.array([[1, 2], [3, 4]], dtype=np.uint8), np.array([[5, 6], [7, 8]], dtype=np.uint8)]
    input_mins = [np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32)]
    input_maxes = [np.array(5.0, dtype=np.float32), np.array(10.0, dtype=np.float32)]

    input_dict = {
        "concat_dim": concat_dim,
        "values": [values[0], values[1]],
        "input_mins": [input_mins[0], input_mins[1]],
        "input_maxes": [input_maxes[0], input_maxes[1]],
        "name": "concat1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    concat_dim = np.array(1, dtype=np.int32)
    values = [np.array([[1, 2], [3, 4]], dtype=np.uint8), np.array([[5, 6], [7, 8]], dtype=np.uint8)]
    input_mins = [np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32)]
    input_maxes = [np.array(5.0, dtype=np.float32), np.array(10.0, dtype=np.float32)]

    input_dict = {
        "concat_dim": concat_dim,
        "values":  [values[0], values[1]],
        "input_mins": [input_mins[0], input_mins[1]],
        "input_maxes": [input_maxes[0], input_maxes[1]],
        "name": "concat2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    concat_dim = np.array(0, dtype=np.int32)
    values = [np.array([1, 2, 3], dtype=np.uint8), np.array([4, 5, 6], dtype=np.uint8), np.array([7, 8, 9], dtype=np.uint8)]
    input_mins = [np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32)]
    input_maxes = [np.array(3.0, dtype=np.float32), np.array(6.0, dtype=np.float32), np.array(9.0, dtype=np.float32)]

    input_dict = {
        "concat_dim": concat_dim,
        "values": [values[0], values[1], values[2]],
        "input_mins": [input_mins[0], input_mins[1], input_mins[2]],
        "input_maxes": [input_maxes[0], input_maxes[1], input_maxes[2]],
        "name": "concat3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    concat_dim = np.array(0, dtype=np.int32)
    values = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.uint8)]
    input_mins = [np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32)]
    input_maxes = [np.array(8.0, dtype=np.float32), np.array(16.0, dtype=np.float32)]

    input_dict = {
        "concat_dim": concat_dim,
        "values": [values[0], values[1]],
        "input_mins": [input_mins[0], input_mins[1]],
        "input_maxes": [input_maxes[0], input_maxes[1]],
        "name": "concat4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    concat_dim = np.array(1, dtype=np.int32)
    values = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.uint8)]
    input_mins = [np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32)]
    input_maxes = [np.array(8.0, dtype=np.float32), np.array(16.0, dtype=np.float32)]

    input_dict = {
        "concat_dim": concat_dim,
        "values": [values[0], values[1]],
        "input_mins": [input_mins[0], input_mins[1]],
        "input_maxes": [input_maxes[0], input_maxes[1]],
        "name": "concat5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    concat_dim = np.array(2, dtype=np.int32)
    values = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.uint8)]
    input_mins = [np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32)]
    input_maxes = [np.array(8.0, dtype=np.float32), np.array(16.0, dtype=np.float32)]

    input_dict = {
        "concat_dim": concat_dim,
        "values": [values[0], values[1]],
        "input_mins": [input_mins[0], input_mins[1]],
        "input_maxes": [input_maxes[0], input_maxes[1]],
        "name": "concat6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    concat_dim = np.array(0, dtype=np.int32)
    values = [np.array([[1, 2, 3]], dtype=np.uint8), np.array([[4, 5, 6]], dtype=np.uint8)]
    input_mins = [np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32)]
    input_maxes = [np.array(3.0, dtype=np.float32), np.array(6.0, dtype=np.float32)]

    input_dict = {
        "concat_dim": concat_dim,
        "values": [values[0], values[1]],
        "input_mins": [input_mins[0], input_mins[1]],
        "input_maxes": [input_maxes[0], input_maxes[1]],
        "name": "concat7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    concat_dim = np.array(1, dtype=np.int32)
    values = [np.array([[1, 2, 3]], dtype=np.uint8), np.array([[4, 5, 6]], dtype=np.uint8)]
    input_mins = [np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32)]
    input_maxes = [np.array(3.0, dtype=np.float32), np.array(6.0, dtype=np.float32)]

    input_dict = {
        "concat_dim": concat_dim,
        "values": [values[0], values[1]],
        "input_mins": [input_mins[0], input_mins[1]],
        "input_maxes": [input_maxes[0], input_maxes[1]],
        "name": "concat8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    concat_dim = np.array(0, dtype=np.int32)
    values = [np.array([1], dtype=np.uint8), np.array([2], dtype=np.uint8), np.array([3], dtype=np.uint8)]
    input_mins = [np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32)]
    input_maxes = [np.array(1.0, dtype=np.float32), np.array(2.0, dtype=np.float32), np.array(3.0, dtype=np.float32)]

    input_dict = {
        "concat_dim": concat_dim,
        "values": [values[0], values[1], values[2]],
        "input_mins": [input_mins[0], input_mins[1], input_mins[2]],
        "input_maxes": [input_maxes[0], input_maxes[1], input_maxes[2]],
        "name": "concat9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    concat_dim = np.array(0, dtype=np.int32)
    values = [np.array([[1, 2], [3, 4]], dtype=np.int8), np.array([[5, 6], [7, 8]], dtype=np.int8)]
    input_mins = [np.array(-5.0, dtype=np.float32), np.array(-10.0, dtype=np.float32)]
    input_maxes = [np.array(5.0, dtype=np.float32), np.array(10.0, dtype=np.float32)]

    input_dict = {
        "concat_dim": concat_dim,
        "values": [values[0], values[1]],
        "input_mins": [input_mins[0], input_mins[1]],
        "input_maxes": [input_maxes[0], input_maxes[1]],
        "name": "concat10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    generated_inputs["tf.raw_ops.QuantizedConcat"] = list_of_inputs
    return list_of_inputs

generated_inputs = {}
tf_raw_ops_quantized_concat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizedConcat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedConcat'.")

check_valid('tf.raw_ops.QuantizedConcat', generated_inputs['tf.raw_ops.QuantizedConcat'], lib="tf", suffix=0)
