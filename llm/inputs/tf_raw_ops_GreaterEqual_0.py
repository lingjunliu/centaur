
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_greater_equal_inputs():
    list_of_inputs = []

    # Input 1: Basic case with integers
    x = np.array([5, 4, 6, 7], dtype=np.int32)
    y = np.array([5, 2, 5, 10], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Broadcasting case with integers
    x = np.array([5, 4, 6, 7], dtype=np.int32)
    y = np.array([5], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Floats
    x = np.array([5.0, 4.0, 6.0, 7.0], dtype=np.float32)
    y = np.array([5.0, 2.0, 5.0, 10.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values
    x = np.array([-5, -4, -6, -7], dtype=np.int32)
    y = np.array([-5, -2, -5, -10], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different shapes
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[2, 1], [4, 3]], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: All True
    x = np.array([5, 5, 5, 5], dtype=np.int32)
    y = np.array([1, 2, 3, 4], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: All False
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([5, 5, 5, 5], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zero values
    x = np.array([0, 0, 0, 0], dtype=np.int32)
    y = np.array([0, 1, -1, 0], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Uint8
    x = np.array([5, 4, 6, 7], dtype=np.uint8)
    y = np.array([5, 2, 5, 10], dtype=np.uint8)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float64, name provided
    x = np.array([5.0, 4.0, 6.0, 7.0], dtype=np.float64)
    y = np.array([5.0, 2.0, 5.0, 10.0], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": "my_greater_equal"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.GreaterEqual"] = tf_raw_ops_greater_equal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.GreaterEqual' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.GreaterEqual'.")

check_valid('tf.raw_ops.GreaterEqual', generated_inputs['tf.raw_ops.GreaterEqual'], lib="tf", suffix=0)
