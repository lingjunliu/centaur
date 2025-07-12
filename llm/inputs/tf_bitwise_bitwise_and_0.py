
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_bitwise_bitwise_and_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    y = np.array([5, 4, 3, 2, 1], dtype=np.int32)
    name = "bitwise_and_1"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([-1, -2, -3], dtype=np.int32)
    y = np.array([1, 2, 3], dtype=np.int32)
    name = "bitwise_and_2"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[1, 2], [3, 4]], dtype=np.int8)
    y = np.array([[5, 6], [7, 8]], dtype=np.int8)
    name = "bitwise_and_3"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([10, 20, 30], dtype=np.int64)
    y = np.array([5, 15, 25], dtype=np.int64)
    name = "bitwise_and_4"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([255, 0, 127], dtype=np.int8)
    y = np.array([127, 255, 0], dtype=np.int8)
    name = "bitwise_and_5"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([1, 2, 3, 4], dtype=np.int16)
    y = np.array([0, 0, 0, 0], dtype=np.int16)
    name = "bitwise_and_6"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([1, 1, 1, 1], dtype=np.int32)
    y = np.array([1, 1, 1, 1], dtype=np.int32)
    name = "bitwise_and_7"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([0, 0, 0, 0], dtype=np.int64)
    y = np.array([1, 1, 1, 1], dtype=np.int64)
    name = "bitwise_and_8"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    y = np.array([[6, 5, 4], [3, 2, 1]], dtype=np.int32)
    name = "bitwise_and_9"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10
    x = np.array([0xFFFFFFFF, 0x00000000], dtype=np.int64)
    y = np.array([0x0000FFFF, 0xFFFFFFFF], dtype=np.int64)
    name = "bitwise_and_10"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.bitwise.bitwise_and"] = tf_bitwise_bitwise_and_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.bitwise.bitwise_and' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.bitwise.bitwise_and'.")

check_valid('tf.bitwise.bitwise_and', generated_inputs['tf.bitwise.bitwise_and'], lib="tf", suffix=0)
