
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_bitwise_left_shift_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([0, 1, 2, 3], dtype=np.int32)
    name = "left_shift_example_1"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([-1, -2, -3, -4], dtype=np.int32)
    y = np.array([0, 1, 2, 3], dtype=np.int32)
    name = "left_shift_example_2"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([1, 2, 3, 4], dtype=np.int64)
    y = np.array([0, 1, 2, 3], dtype=np.int64)
    name = "left_shift_example_3"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([-1, -2, -3, -4], dtype=np.int64)
    y = np.array([0, 1, 2, 3], dtype=np.int64)
    name = "left_shift_example_4"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([1, 2, 3, 4], dtype=np.int8)
    y = np.array([0, 1, 2, 3], dtype=np.int8)
    name = "left_shift_example_5"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([-1, -2, -3, -4], dtype=np.int8)
    y = np.array([0, 1, 2, 3], dtype=np.int8)
    name = "left_shift_example_6"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - Multidimensional array
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[0, 1], [2, 3]], dtype=np.int32)
    name = "left_shift_example_7"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 - Larger shift values
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([5, 10, 15, 20], dtype=np.int32)
    name = "left_shift_example_8"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.bitwise.left_shift"] = tf_bitwise_left_shift_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.bitwise.left_shift' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.bitwise.left_shift'.")

check_valid('tf.bitwise.left_shift', generated_inputs['tf.bitwise.left_shift'], lib="tf", suffix=0)
