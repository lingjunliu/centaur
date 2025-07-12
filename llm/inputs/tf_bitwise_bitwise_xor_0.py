
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_bitwise_bitwise_xor_inputs():
    list_of_inputs = []

    # Input 1: Basic int32
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([4, 3, 2, 1], dtype=np.int32)
    name = "xor_basic_int32"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: uint8 with different values
    x = np.array([0, 255, 128, 64], dtype=np.uint8)
    y = np.array([255, 0, 64, 128], dtype=np.uint8)
    name = "xor_uint8_diff"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int16 with negative values
    x = np.array([-1, -2, 3, 4], dtype=np.int16)
    y = np.array([4, 3, -2, -1], dtype=np.int16)
    name = "xor_int16_neg"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, int32
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[4, 3], [2, 1]], dtype=np.int32)
    name = "xor_2d_int32"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, int64
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    y = np.array([[[8, 7], [6, 5]], [[4, 3], [2, 1]]], dtype=np.int64)
    name = "xor_3d_int64"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int64 with large values
    x = np.array([2147483647, 1], dtype=np.int64)
    y = np.array([1, 2147483647], dtype=np.int64)
    name = "xor_int64_large"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array, int8
    x = np.array([10, -5, 20, -10], dtype=np.int8)
    y = np.array([-10, 20, -5, 10], dtype=np.int8)
    name = "xor_int8"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array, int16
    x = np.array([[1000, 2000], [3000, 4000]], dtype=np.int16)
    y = np.array([[4000, 3000], [2000, 1000]], dtype=np.int16)
    name = "xor_2d_int16"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Identical arrays
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([1, 2, 3, 4], dtype=np.int32)
    name = "xor_identical"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: All zeros
    x = np.array([0, 0, 0, 0], dtype=np.int32)
    y = np.array([0, 0, 0, 0], dtype=np.int32)
    name = "xor_zeros"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.bitwise.bitwise_xor"] = tf_bitwise_bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.bitwise.bitwise_xor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.bitwise.bitwise_xor'.")

check_valid('tf.bitwise.bitwise_xor', generated_inputs['tf.bitwise.bitwise_xor'], lib="tf", suffix=0)
