
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FloorDiv_inputs():
    list_of_inputs = []

    # Input 1: int32
    x = np.array([10, 20, 30], dtype=np.int32)
    y = np.array([3, 7, 2], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32
    x = np.array([10.5, 20.7, 30.9], dtype=np.float32)
    y = np.array([3.2, 7.1, 2.8], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int64 with negative values
    x = np.array([-10, -20, 30], dtype=np.int64)
    y = np.array([3, -7, 2], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64 with broadcasting
    x = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
    y = np.array([2.0, 4.0], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64
    x = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
    y = np.array([1-1j, 2+0j, 1+1j], dtype=np.complex64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: uint8
    x = np.array([100, 200, 255], dtype=np.uint8)
    y = np.array([10, 20, 5], dtype=np.uint8)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int16
    x = np.array([1000, 2000, 3000], dtype=np.int16)
    y = np.array([300, 700, 200], dtype=np.int16)
    input_dict = {"x": x, "y": y, "name": "division"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array int32
    x = np.array([[10, 20], [30, 40]], dtype=np.int32)
    y = np.array([[3, 7], [2, 5]], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.FloorDiv"] = tf_raw_ops_FloorDiv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FloorDiv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FloorDiv'.")

check_valid('tf.raw_ops.FloorDiv', generated_inputs['tf.raw_ops.FloorDiv'], lib="tf", suffix=0)
