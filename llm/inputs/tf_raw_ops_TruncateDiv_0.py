
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_TruncateDiv_inputs():
    list_of_inputs = []

    # Input 1: int32, basic division
    x = np.array([10, 20, 30], dtype=np.int32)
    y = np.array([2, 5, 10], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, division with truncation
    x = np.array([7.5, -7.5, 10.0], dtype=np.float32)
    y = np.array([2.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int64, division with larger numbers
    x = np.array([10000000000, 20000000000, -30000000000], dtype=np.int64)
    y = np.array([2, 5, 10], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: uint8, division with unsigned integers
    x = np.array([255, 128, 64], dtype=np.uint8)
    y = np.array([5, 2, 1], dtype=np.uint8)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D int32 array
    x = np.array([[10, 20], [30, 40]], dtype=np.int32)
    y = np.array([[2, 5], [10, 8]], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting example (int32)
    x = np.array([[10, 20], [30, 40]], dtype=np.int32)
    y = np.array([2, 5], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int8 with negative values
    x = np.array([-10, 20, -30], dtype=np.int8)
    y = np.array([2, -5, 10], dtype=np.int8)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.TruncateDiv"] = tf_raw_ops_TruncateDiv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.TruncateDiv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.TruncateDiv'.")

check_valid('tf.raw_ops.TruncateDiv', generated_inputs['tf.raw_ops.TruncateDiv'], lib="tf", suffix=0)
