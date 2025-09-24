
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_floormod_inputs():
    list_of_inputs = []

    # Input 1: int32
    x = np.array([5, 3, 7, 2], dtype=np.int32)
    y = np.array([2, 4, 3, 5], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "floormod_int32_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int64
    x = np.array([-5, 3, -7, 2], dtype=np.int64)
    y = np.array([2, -4, 3, -5], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": "floormod_int64_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32
    x = np.array([5.0, 3.0, 7.0, 2.0], dtype=np.float32)
    y = np.array([2.0, 4.0, 3.0, 5.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "floormod_float32_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64
    x = np.array([-5.0, 3.0, -7.0, 2.0], dtype=np.float64)
    y = np.array([2.0, -4.0, 3.0, -5.0], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": "floormod_float64_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: uint8
    x = np.array([5, 3, 7, 2], dtype=np.uint8)
    y = np.array([2, 4, 3, 5], dtype=np.uint8)
    input_dict = {"x": x, "y": y, "name": "floormod_uint8_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int16, different shapes
    x = np.array([[5, 3], [7, 2]], dtype=np.int16)
    y = np.array([[2, 4], [3, 5]], dtype=np.int16)
    input_dict = {"x": x, "y": y, "name": "floormod_int16_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int8, negative values, scalar y
    x = np.array([-5, 3, -7, 2], dtype=np.int8)
    y = np.array(-2, dtype=np.int8)
    input_dict = {"x": x, "y": y, "name": "floormod_int8_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float16
    x = np.array([5.0, 3.0, 7.0, 2.0], dtype=np.float16)
    y = np.array([2.0, 4.0, 3.0, 5.0], dtype=np.float16)
    input_dict = {"x": x, "y": y, "name": "floormod_float16_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.FloorMod"] = tf_raw_ops_floormod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FloorMod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FloorMod'.")

check_valid('tf.raw_ops.FloorMod', generated_inputs['tf.raw_ops.FloorMod'], lib="tf", suffix=0)
