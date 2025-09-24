
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_floordiv_inputs():
    list_of_inputs = []

    # Input 1: Integer tensors
    x = np.array([10, 20, 30], dtype=np.int32)
    y = np.array([3, 7, 2], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "floor_div_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float tensors
    x = np.array([10.5, 20.2, 30.7], dtype=np.float32)
    y = np.array([3.1, 7.0, 2.5], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "floor_div_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Broadcasting
    x = np.array([[10, 20], [30, 40]], dtype=np.int32)
    y = np.array([2, 5], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "floor_div_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative numbers
    x = np.array([-10, -20, 30], dtype=np.int32)
    y = np.array([3, -7, 2], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "floor_div_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different integer types
    x = np.array([10, 20, 30], dtype=np.int64)
    y = np.array([3, 7, 2], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": "floor_div_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multi-dimensional array
    x = np.array([[[10, 20], [30, 40]], [[50, 60], [70, 80]]], dtype=np.int32)
    y = np.array([[[2, 5], [3, 7]], [[5, 2], [7, 3]]], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "floor_div_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint8
    x = np.array([255, 128, 64], dtype=np.uint8)
    y = np.array([10, 5, 2], dtype=np.uint8)
    input_dict = {"x": x, "y": y, "name": "floor_div_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64
    x = np.array([10.5, 20.2, 30.7], dtype=np.float64)
    y = np.array([3.1, 7.0, 2.5], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": "floor_div_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: half
    x = np.array([10.5, 20.2, 30.7], dtype=np.float16)
    y = np.array([3.1, 7.0, 2.5], dtype=np.float16)
    input_dict = {"x": x, "y": y, "name": "floor_div_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.FloorDiv"] = tf_raw_ops_floordiv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FloorDiv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FloorDiv'.")

check_valid('tf.raw_ops.FloorDiv', generated_inputs['tf.raw_ops.FloorDiv'], lib="tf", suffix=0)
