
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_bitwise_bitwise_or_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([0, 5, 3, 14], dtype=np.int32)
    y = np.array([5, 0, 7, 11], dtype=np.int32)
    name = "test_or_1"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([-1, -2, -3, -4], dtype=np.int64)
    y = np.array([1, 2, 3, 4], dtype=np.int64)
    name = "test_or_2"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    y = np.array([[5, 6], [7, 8]], dtype=np.uint8)
    name = "test_or_3"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16)
    y = np.array([[[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.int16)
    name = "test_or_4"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([10, 20, 30, 40], dtype=np.uint32)
    y = np.array([5, 10, 15, 20], dtype=np.uint32)
    name = "test_or_5"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([0xFFFFFFFF], dtype=np.int32)
    y = np.array([0x0000000F], dtype=np.int32)
    name = "test_or_6"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([255, 255], dtype=np.uint8)
    y = np.array([0, 0], dtype=np.uint8)
    name = "test_or_7"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([1, 2, 3, 4, 5], dtype=np.int8)
    y = np.array([5, 4, 3, 2, 1], dtype=np.int8)
    name = "test_or_8"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([65535], dtype=np.uint16)
    y = np.array([1], dtype=np.uint16)
    name = "test_or_9"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([[1,0],[0,1]], dtype=np.int32)
    y = np.array([[0,1],[1,0]], dtype=np.int32)
    name = "test_or_10"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
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
