
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_truncate_mod_inputs():
    list_of_inputs = []

    # Input 1: Basic integer input
    x = np.array([10, 15, 20], dtype=np.int32)
    y = np.array([3, 4, 7], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative integer input
    x = np.array([-10, -15, -20], dtype=np.int32)
    y = np.array([3, -4, 7], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float input
    x = np.array([10.5, 15.2, 20.7], dtype=np.float32)
    y = np.array([3.0, 4.0, 7.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative float input
    x = np.array([-10.5, -15.2, -20.7], dtype=np.float32)
    y = np.array([3.0, -4.0, 7.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting input
    x = np.array([[10, 15], [20, 25]], dtype=np.int32)
    y = np.array([3, 4], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting input (different shapes)
    x = np.array([10, 15, 20], dtype=np.int32)
    y = np.array(5, dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int64 input
    x = np.array([10, 15, 20], dtype=np.int64)
    y = np.array([3, 4, 7], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64 input
    x = np.array([10.5, 15.2, 20.7], dtype=np.float64)
    y = np.array([3.0, 4.0, 7.0], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: multi-dimensional array
    x = np.array([[[10, 11], [12, 13]], [[14, 15], [16, 17]]], dtype=np.int32)
    y = np.array([3, 4], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32 with large values
    x = np.array([1e9, 2e9, 3e9], dtype=np.float32)
    y = np.array([10, 20, 30], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.TruncateMod"] = tf_raw_ops_truncate_mod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.TruncateMod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.TruncateMod'.")

check_valid('tf.raw_ops.TruncateMod', generated_inputs['tf.raw_ops.TruncateMod'], lib="tf", suffix=0)
