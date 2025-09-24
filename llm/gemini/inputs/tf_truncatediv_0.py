
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_truncatediv_inputs():
    list_of_inputs = []

    # Input 1: Basic integer division
    x = np.array([10, 20, -30], dtype=np.int32)
    y = np.array([3, 4, 5], dtype=np.int32)
    name = "basic_division"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Division with zeros (y=1 to avoid division by zero)
    x = np.array([5, 0, -5], dtype=np.int32)
    y = np.array([2, 1, 1], dtype=np.int32)
    name = "division_with_ones"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Floating-point division
    x = np.array([7.5, -9.2, 11.8], dtype=np.float32)
    y = np.array([2.0, 3.0, -4.0], dtype=np.float32)
    name = "float_division"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-dimensional array
    x = np.array([[10, 20], [30, 40]], dtype=np.int32)
    y = np.array([[2, 5], [6, 8]], dtype=np.int32)
    name = "multi_dimensional"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting
    x = np.array([10, 20, 30], dtype=np.int32)
    y = np.array([2], dtype=np.int32)
    name = "broadcasting"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different integer types
    x = np.array([10, 20, 30], dtype=np.int64)
    y = np.array([2, 5, 6], dtype=np.int64)
    name = "int64_division"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative division with floats
    x = np.array([-7.5, -9.2, -11.8], dtype=np.float32)
    y = np.array([2.0, -3.0, 4.0], dtype=np.float32)
    name = "negative_float_division"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger integer values
    x = np.array([1000, 2000, -3000], dtype=np.int32)
    y = np.array([3, 4, 5], dtype=np.int32)
    name = "large_int_division"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: half
    x = np.array([7.5, -9.2, 11.8], dtype=np.float16)
    y = np.array([2.0, 3.0, -4.0], dtype=np.float16)
    name = "half_division"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int8
    x = np.array([10, 20, -30], dtype=np.int8)
    y = np.array([3, 4, 5], dtype=np.int8)
    name = "int8_division"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.truncatediv"] = tf_truncatediv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.truncatediv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.truncatediv'.")

check_valid('tf.truncatediv', generated_inputs['tf.truncatediv'], lib="tf", suffix=0)
