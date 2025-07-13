
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FloorDiv_inputs():
    list_of_inputs = []

    # Input 1: int32, 1D arrays
    x = np.array([10, 20, 30], dtype=np.int32)
    y = np.array([3, 7, 2], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, 2D arrays
    x = np.array([[10.5, 20.3], [30.1, 40.8]], dtype=np.float32)
    y = np.array([[3.0, 7.0], [2.0, 5.0]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "floor_div_example_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int64, scalar values
    x = np.array(100, dtype=np.int64)
    y = np.array(7, dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex64, 1D arrays
    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    y = np.array([1j, 1+0j, 1-1j], dtype=np.complex64)
    input_dict = {"x": x, "y": y, "name": "floor_div_example_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: uint8, 2D arrays
    x = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    y = np.array([[3, 7], [2, 5]], dtype=np.uint8)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int8, 1D arrays with negative values
    x = np.array([-10, 20, -30], dtype=np.int8)
    y = np.array([3, -7, 2], dtype=np.int8)
    input_dict = {"x": x, "y": y, "name": "floor_div_example_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: float16, 2D arrays
    x = np.array([[10.5, 20.3], [30.1, 40.8]], dtype=np.float16)
    y = np.array([[3.0, 7.0], [2.0, 5.0]], dtype=np.float16)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int32, 3D arrays
    x = np.array([[[10, 20], [30, 40]], [[50, 60], [70, 80]]], dtype=np.int32)
    y = np.array([[[3, 7], [2, 5]], [[4, 6], [8, 9]]], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint64, large values
    x = np.array([18446744073709551615, 9223372036854775807], dtype=np.uint64)
    y = np.array([2, 3], dtype=np.uint64)
    input_dict = {"x": x, "y": y, "name": "floor_div_example_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex128
    x = np.array([10+5j, 20-3j], dtype=np.complex128)
    y = np.array([2+1j, 4-2j], dtype=np.complex128)
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
