
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
    y = np.array([0.5, 2.0, 1.5], dtype=np.float32)
    name = "basic_division"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Broadcasting with different shapes
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0], dtype=np.float32)
    name = "broadcasting_division"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Integer division
    x = np.array([4, 6, 8], dtype=np.int32)
    y = np.array([2, 3, 4], dtype=np.int32)
    name = "integer_division"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Division with negative numbers
    x = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    y = np.array([0.5, -2.0, 1.5], dtype=np.float32)
    name = "negative_division"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Division with complex numbers
    x = np.array([1 + 1j, 2 - 2j, 3 + 0j], dtype=np.complex64)
    y = np.array([1j, 2 + 0j, 1 - 1j], dtype=np.complex64)
    name = "complex_division"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multi-dimensional array division
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    name = "multi_dimensional_division"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint8 division
    x = np.array([10, 20, 30], dtype=np.uint8)
    y = np.array([2, 5, 10], dtype=np.uint8)
    name = "uint8_division"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int64 division
    x = np.array([10000000000, 20000000000, 30000000000], dtype=np.int64)
    y = np.array([2, 5, 10], dtype=np.int64)
    name = "int64_division"
    input_dict = {"x": x, "y": y, "name": name}
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
