
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_truncate_div_inputs():
    list_of_inputs = []

    # Input 1: Basic integer division
    x = np.array([10, 20, 30], dtype=np.int32)
    y = np.array([2, 3, 4], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float division
    x = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    y = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "float_division"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative numbers
    x = np.array([-10, -20, -30], dtype=np.int32)
    y = np.array([2, 3, 4], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "negative_numbers"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Mixed positive and negative
    x = np.array([-10, 20, -30], dtype=np.int32)
    y = np.array([2, -3, 4], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "mixed_numbers"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Division by one
    x = np.array([10, 20, 30], dtype=np.int32)
    y = np.array([1, 1, 1], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "division_by_one"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Two-dimensional array
    x = np.array([[10, 20], [30, 40]], dtype=np.int32)
    y = np.array([[2, 3], [4, 5]], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "two_dimensional"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Division resulting in zero
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([10, 20, 30], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "division_to_zero"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex numbers
    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    y = np.array([1j, 2j, 3j], dtype=np.complex64)
    input_dict = {"x": x, "y": y, "name": "complex_numbers"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint8
    x = np.array([10, 20, 30], dtype=np.uint8)
    y = np.array([2, 3, 4], dtype=np.uint8)
    input_dict = {"x": x, "y": y, "name": "uint8_division"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: bfloat16
    x = np.array([tf.constant(10.0, dtype=tf.bfloat16).numpy(), tf.constant(20.0, dtype=tf.bfloat16).numpy(), tf.constant(30.0, dtype=tf.bfloat16).numpy()], dtype=tf.bfloat16.as_numpy_dtype())
    y = np.array([tf.constant(2.0, dtype=tf.bfloat16).numpy(), tf.constant(3.0, dtype=tf.bfloat16).numpy(), tf.constant(4.0, dtype=tf.bfloat16).numpy()], dtype=tf.bfloat16.as_numpy_dtype())

    input_dict = {"x": x, "y": y, "name": "bfloat16_division"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.TruncateDiv"] = tf_raw_ops_truncate_div_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.TruncateDiv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.TruncateDiv'.")

check_valid('tf.raw_ops.TruncateDiv', generated_inputs['tf.raw_ops.TruncateDiv'], lib="tf", suffix=0)
