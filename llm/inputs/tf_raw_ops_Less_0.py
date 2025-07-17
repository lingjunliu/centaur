
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_less_inputs():
    list_of_inputs = []

    # Input 1: Basic integers
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([2, 2, 2], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "less_int"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic floats
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 1.0, 4.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "less_float"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Broadcasting with integers
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([2], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "less_int_broadcast"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting with floats
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "less_float_broadcast"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative integers
    x = np.array([-1, -2, -3], dtype=np.int32)
    y = np.array([-2, -2, -2], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "less_int_negative"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative floats
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    y = np.array([-2.0, -1.0, -4.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "less_float_negative"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multi-dimensional integers
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[2, 1], [4, 3]], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "less_int_multi"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multi-dimensional floats
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[2.0, 1.0], [4.0, 3.0]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "less_float_multi"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different integer types
    x = np.array([1, 2, 3], dtype=np.int64)
    y = np.array([2, 1, 4], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": "less_int64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different float types
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([2.0, 1.0, 4.0], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": "less_float64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Less"] = tf_raw_ops_less_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Less' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Less'.")

check_valid('tf.raw_ops.Less', generated_inputs['tf.raw_ops.Less'], lib="tf", suffix=0)
