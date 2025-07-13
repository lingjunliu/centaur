
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_floormod_inputs():
    list_of_inputs = []

    # Input 1: Basic integer division
    x = np.array([7, 10, -5], dtype=np.int32)
    y = np.array([3, 4, 2], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Broadcasting with integers
    x = np.array([[7, 10], [-5, 12]], dtype=np.int32)
    y = np.array([3, 4], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Floating-point numbers
    x = np.array([7.5, 10.2, -5.8], dtype=np.float32)
    y = np.array([3.0, 4.0, 2.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting with floats
    x = np.array([[7.5, 10.2], [-5.8, 12.1]], dtype=np.float32)
    y = np.array([3.0, 4.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative numbers
    x = np.array([-7, 10, -5], dtype=np.int32)
    y = np.array([3, -4, 2], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different shapes (broadcasting)
    x = np.array([[-7, 10, -5]], dtype=np.int32)
    y = np.array([3, -4, 2], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint64 values
    x = np.array([18446744073709551615, 10, 5], dtype=np.uint64)
    y = np.array([3, 4, 2], dtype=np.uint64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 64-bit integers
    x = np.array([7, 10, -5], dtype=np.int64)
    y = np.array([3, 4, 2], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Mixed positive and negative floats
    x = np.array([7.5, -10.2, 5.8, -12.1], dtype=np.float64)
    y = np.array([-3.0, 4.0, -2.0, 5.0], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Zero values
    x = np.array([7, 0, -5], dtype=np.int32)
    y = np.array([3, 4, 0], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
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
