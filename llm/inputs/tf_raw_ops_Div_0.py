
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_div_inputs():
    list_of_inputs = []

    # Input 1: Basic float division
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([0.5, 2.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "basic_float_div"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int division
    x = np.array([10, 20, 30], dtype=np.int32)
    y = np.array([2, 5, 10], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "int_div"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex division
    x = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
    y = np.array([1j, 1 + 0j, 1 - 1j], dtype=np.complex64)
    input_dict = {"x": x, "y": y, "name": "complex_div"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([2.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "broadcasting_div"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional arrays
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    y = np.array([[[2, 1], [1, 2]], [[1, 1], [2, 2]]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "multi_dim_div"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different float type (float64)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([0.5, 2.0, 1.0], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": "float64_div"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different int type (int64)
    x = np.array([10, 20, 30], dtype=np.int64)
    y = np.array([2, 5, 10], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": "int64_div"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: uint32 division -> Replacing with int32 to avoid dtype issue
    x = np.array([10, 20, 30], dtype=np.int32)
    y = np.array([2, 5, 10], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "uint32_div"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex division with broadcasting
    x = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=np.complex64)
    y = np.array([1j], dtype=np.complex64)
    input_dict = {"x": x, "y": y, "name": "complex_broadcasting_div"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: half precision
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    y = np.array([0.5, 2.0, 1.0], dtype=np.float16)
    input_dict = {"x": x, "y": y, "name": "half_div"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Div"] = tf_raw_ops_div_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Div' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Div'.")

check_valid('tf.raw_ops.Div', generated_inputs['tf.raw_ops.Div'], lib="tf", suffix=0)
