
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_bitwise_right_shift_inputs():
    list_of_inputs = []

    # Input 1: int32, simple case
    x = np.array([16, 32, 64], dtype=np.int32)
    y = np.array([1, 2, 3], dtype=np.int32)
    name = "right_shift_1"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int64, with negative values
    x = np.array([-16, -32, -64], dtype=np.int64)
    y = np.array([1, 2, 3], dtype=np.int64)
    name = "right_shift_2"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: uint8, with zeros
    x = np.array([0, 1, 2], dtype=np.uint8)
    y = np.array([0, 1, 0], dtype=np.uint8)
    name = "right_shift_3"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int16, different shapes
    x = np.array([[16, 32], [64, 128]], dtype=np.int16)
    y = np.array([[1, 2], [3, 4]], dtype=np.int16)
    name = "right_shift_4"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int32, shifting by 0
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([0, 0, 0], dtype=np.int32)
    name = "right_shift_6"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int32, multi-dimensional array
    x = np.array([[[16, 32], [64, 128]], [[256, 512], [1024, 2048]]], dtype=np.int32)
    y = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    name = "right_shift_8"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int64, zeros in x
    x = np.array([0, 0, 0], dtype=np.int64)
    y = np.array([1, 2, 3], dtype=np.int64)
    name = "right_shift_9"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int8
    x = np.array([1, 2, 3], dtype=np.int8)
    y = np.array([1, 2, 3], dtype=np.int8)
    name = "right_shift_11"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int8, negative values
    x = np.array([-1, -2, -3], dtype=np.int8)
    y = np.array([1, 1, 1], dtype=np.int8)
    name = "right_shift_12"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint8
    x = np.array([1, 2, 3], dtype=np.uint8)
    y = np.array([1, 2, 3], dtype=np.uint8)
    name = "right_shift_13"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.bitwise.right_shift"] = tf_bitwise_right_shift_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.bitwise.right_shift' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.bitwise.right_shift'.")

check_valid('tf.bitwise.right_shift', generated_inputs['tf.bitwise.right_shift'], lib="tf", suffix=0)
