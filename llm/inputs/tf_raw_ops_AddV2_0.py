
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_addv2_inputs():
    list_of_inputs = []

    # Input 1: Simple addition of two scalars
    x = np.array(5, dtype=np.float32)
    y = np.array(3, dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Addition of two 1D arrays
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Addition of two 2D arrays
    x = np.array([[1, 2], [3, 4]], dtype=np.float64)
    y = np.array([[5, 6], [7, 8]], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Addition with negative values
    x = np.array([-1, -2, -3], dtype=np.int64)
    y = np.array([1, 2, 3], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Addition with different data types (half)
    x = np.array([1, 2, 3], dtype=np.float16)
    y = np.array([4, 5, 6], dtype=np.float16)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Addition with complex numbers
    x = np.array([1 + 1j, 2 + 2j], dtype=np.complex64)
    y = np.array([3 + 3j, 4 + 4j], dtype=np.complex64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Addition with uint8
    x = np.array([1, 2, 3], dtype=np.uint8)
    y = np.array([4, 5, 6], dtype=np.uint8)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Addition with int8
    x = np.array([-1, 2, -3], dtype=np.int8)
    y = np.array([4, -5, 6], dtype=np.int8)
    input_dict = {"x": x, "y": y, "name": "int8_add"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Addition of two 3D arrays
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    y = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger numbers
    x = np.array([1000, 2000], dtype=np.int32)
    y = np.array([3000, 4000], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AddV2"] = tf_raw_ops_addv2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.AddV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AddV2'.")

check_valid('tf.raw_ops.AddV2', generated_inputs['tf.raw_ops.AddV2'], lib="tf", suffix=0)
