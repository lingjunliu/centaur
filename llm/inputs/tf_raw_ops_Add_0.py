
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_add_inputs():
    list_of_inputs = []

    # Input 1: Basic addition of two scalars
    x = np.array(5, dtype=np.int32)
    y = np.array(3, dtype=np.int32)
    name = "add_scalar"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Addition of two 1D arrays
    x = np.array([1, 2, 3], dtype=np.float32)
    y = np.array([4, 5, 6], dtype=np.float32)
    name = "add_1d_array"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Addition of two 2D arrays
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    y = np.array([[5, 6], [7, 8]], dtype=np.int64)
    name = "add_2d_array"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Addition with negative numbers
    x = np.array([-1, -2, -3], dtype=np.float64)
    y = np.array([4, 5, -6], dtype=np.float64)
    name = "add_negative"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Addition with broadcasting (scalar + array)
    x = np.array(2, dtype=np.int16)
    y = np.array([1, 2, 3], dtype=np.int16)
    name = "add_broadcast_scalar"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Addition with complex numbers
    x = np.array([1+1j, 2+2j], dtype=np.complex64)
    y = np.array([3+3j, 4+4j], dtype=np.complex64)
    name = "add_complex"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Addition with different shapes (broadcasting)
    x = np.array([[1, 2, 3]], dtype=np.float32)
    y = np.array([4, 5, 6], dtype=np.float32)
    name = "add_broadcast_shape"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Addition with unsigned integers
    x = np.array([1, 2, 3], dtype=np.uint8)
    y = np.array([4, 5, 6], dtype=np.uint8)
    name = "add_uint8"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Addition with bfloat16
    x = np.array([1, 2, 3], dtype=np.float16)
    y = np.array([4, 5, 6], dtype=np.float16)
    name = "add_bfloat16"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Addition with a 3D array
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    y = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int32)
    name = "add_3d_array"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Add"] = tf_raw_ops_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Add' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Add'.")

check_valid('tf.raw_ops.Add', generated_inputs['tf.raw_ops.Add'], lib="tf", suffix=0)
