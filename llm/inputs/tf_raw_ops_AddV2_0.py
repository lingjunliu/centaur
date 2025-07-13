
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_addv2_inputs():
    list_of_inputs = []

    # Input 1: Basic addition with integers
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Addition with floats
    x = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    y = np.array([4.5, 5.5, 6.5], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "float_add"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Addition with negative values
    x = np.array([-1, -2, -3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array addition
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[5, 6], [7, 8]], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array addition with floats
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    y = np.array([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "3d_add"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Addition with uint8
    x = np.array([1, 2, 3], dtype=np.uint8)
    y = np.array([4, 5, 6], dtype=np.uint8)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Addition with int8
    x = np.array([-1, 2, -3], dtype=np.int8)
    y = np.array([4, -5, 6], dtype=np.int8)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Addition with complex64
    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    y = np.array([4+4j, 5+5j, 6+6j], dtype=np.complex64)
    input_dict = {"x": x, "y": y, "name": "complex_add"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Addition with complex128
    x = np.array([1+1j, 2+2j], dtype=np.complex128)
    y = np.array([3+3j, 4+4j], dtype=np.complex128)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Addition with float16
    x = np.array([1, 2, 3], dtype=np.float16)
    y = np.array([4, 5, 6], dtype=np.float16)
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
