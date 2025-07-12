
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_bitwise_bitwise_or_inputs():
    list_of_inputs = []

    # Input 1: Basic case with int32
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    y = np.array([5, 4, 3, 2, 1], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different values with uint8
    x = np.array([255, 128, 64, 32, 16], dtype=np.uint8)
    y = np.array([1, 2, 4, 8, 16], dtype=np.uint8)
    input_dict = {"x": x, "y": y, "name": "uint8_or"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values with int16
    x = np.array([-1, -2, -3, -4, -5], dtype=np.int16)
    y = np.array([5, 4, 3, 2, 1], dtype=np.int16)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-dimensional array with int64
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    y = np.array([[5, 6], [7, 8]], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": "int64_or"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different shapes (but broadcastable) with int8
    x = np.array([1, 2, 3], dtype=np.int8)
    y = np.array([[1], [2], [3]], dtype=np.int8)
    input_dict = {"x": x, "y": y, "name": "int8_or"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Mix of positive and negative with int32
    x = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    y = np.array([10, 5, 0, -5, -10], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: all ones
    x = np.array([1, 1, 1, 1, 1], dtype=np.int32)
    y = np.array([1, 1, 1, 1, 1], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.bitwise.bitwise_or"] = tf_bitwise_bitwise_or_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.bitwise.bitwise_or' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.bitwise.bitwise_or'.")

check_valid('tf.bitwise.bitwise_or', generated_inputs['tf.bitwise.bitwise_or'], lib="tf", suffix=0)
