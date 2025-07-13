
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
    y = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    name = "float_div"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer division
    x = np.array([4, 6, 8], dtype=np.int32)
    y = np.array([2, 3, 4], dtype=np.int32)
    name = "int_div"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Broadcasting example
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([1.0, 2.0], dtype=np.float32)
    name = "broadcast_div"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different data type (bfloat16)
    x = np.array([5.0, 10.0], dtype=np.float32).astype(np.float16)
    y = np.array([2.0, 5.0], dtype=np.float32).astype(np.float16)
    name = "bfloat16_div"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Higher dimensional array
    x = np.random.rand(2, 3, 4).astype(np.float64)
    y = np.random.rand(2, 3, 4).astype(np.float64)
    name = "high_dim_div"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative values
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    y = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    name = "negative_div"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex numbers
    x = np.array([1 + 1j, 2 + 2j], dtype=np.complex64)
    y = np.array([1 - 1j, 2 - 2j], dtype=np.complex64)
    name = "complex_div"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: unsigned int
    x = np.array([4, 6, 8], dtype=np.uint32)
    y = np.array([2, 3, 4], dtype=np.uint32)
    name = "uint_div"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed positive and negative integers
    x = np.array([-4, 6, -8], dtype=np.int32)
    y = np.array([2, -3, 4], dtype=np.int32)
    name = "mixed_int_div"
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
