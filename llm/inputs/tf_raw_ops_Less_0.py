
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
    name = "less_basic_int"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Floats with broadcasting
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0], dtype=np.float32)
    name = "less_float_broadcast"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative integers
    x = np.array([-1, -2, -3], dtype=np.int32)
    y = np.array([0, -1, -2], dtype=np.int32)
    name = "less_negative_int"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array of integers
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[2, 3], [4, 5]], dtype=np.int32)
    name = "less_2d_int"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array of floats
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[0.5, 2.5], [3.5, 5.0]], dtype=np.float32)
    name = "less_2d_float"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6: Different shapes with broadcasting (int64)
    x = np.array([[1, 2, 3]], dtype=np.int64)
    y = np.array([2], dtype=np.int64)
    name = "less_diff_shape_int64"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32 with name
    x = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    y = np.array([2.0, 3.0, 2.0], dtype=np.float32)
    name = "test_less_with_name"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: uint16
    x = np.array([1, 2, 3], dtype=np.uint16)
    y = np.array([2, 1, 4], dtype=np.uint16)
    name = "less_uint16"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: half
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    y = np.array([2.0, 1.0, 4.0], dtype=np.float16)
    name = "less_half"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed positive and negative floats
    x = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    y = np.array([0.0, 1.0, -2.0], dtype=np.float32)
    name = "less_mixed_floats"
    input_dict = {"x": x, "y": y, "name": name}
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
