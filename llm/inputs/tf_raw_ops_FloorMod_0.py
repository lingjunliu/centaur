
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FloorMod_inputs():
    list_of_inputs = []

    # Input 1: int32, basic case
    x = np.array([5, 7, 10], dtype=np.int32)
    y = np.array([2, 3, 4], dtype=np.int32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int32), "y": tf.convert_to_tensor(y, dtype=tf.int32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int64, with negative numbers
    x = np.array([-5, 7, -10], dtype=np.int64)
    y = np.array([2, -3, 4], dtype=np.int64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int64), "y": tf.convert_to_tensor(y, dtype=tf.int64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, basic case
    x = np.array([5.5, 7.2, 10.8], dtype=np.float32)
    y = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "y": tf.convert_to_tensor(y, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, with negative numbers
    x = np.array([-5.5, 7.2, -10.8], dtype=np.float64)
    y = np.array([2.0, -3.0, 4.0], dtype=np.float64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float64), "y": tf.convert_to_tensor(y, dtype=tf.float64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: uint8, basic case
    x = np.array([5, 7, 10], dtype=np.uint8)
    y = np.array([2, 3, 4], dtype=np.uint8)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.uint8), "y": tf.convert_to_tensor(y, dtype=tf.uint8), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D int32 array
    x = np.array([[5, 7], [10, 12]], dtype=np.int32)
    y = np.array([[2, 3], [4, 5]], dtype=np.int32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int32), "y": tf.convert_to_tensor(y, dtype=tf.int32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float64 array with broadcasting
    x = np.array([[5.5, 7.2], [10.8, 12.5]], dtype=np.float64)
    y = np.array([2.0, 3.0], dtype=np.float64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float64), "y": tf.convert_to_tensor(y, dtype=tf.float64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D int64 array with negative numbers
    x = np.array([[[1, 2], [3, 4]], [[-5, -6], [-7, -8]]], dtype=np.int64)
    y = np.array([[[2, -1], [1, 2]], [[-1, 2], [2, -1]]], dtype=np.int64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int64), "y": tf.convert_to_tensor(y, dtype=tf.int64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: bfloat16
    x = np.array([5.5, 7.2, 10.8], dtype=np.float16)
    y = np.array([2.0, 3.0, 4.0], dtype=np.float16)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.bfloat16), "y": tf.convert_to_tensor(y, dtype=tf.bfloat16), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint32
    x = np.array([5, 7, 10], dtype=np.uint32)
    y = np.array([2, 3, 4], dtype=np.uint32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.uint32), "y": tf.convert_to_tensor(y, dtype=tf.uint32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.FloorMod"] = tf_raw_ops_FloorMod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FloorMod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FloorMod'.")

check_valid('tf.raw_ops.FloorMod', generated_inputs['tf.raw_ops.FloorMod'], lib="tf", suffix=0)
