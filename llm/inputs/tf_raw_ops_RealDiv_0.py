
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_realdiv_inputs():
    list_of_inputs = []

    # Input 1: float32, basic case
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([0.5, 2.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, different values, broadcasting
    x = np.array([1.0, 2.0], dtype=np.float64)
    y = np.array(2.0, dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: bfloat16, small values
    x = np.array([0.1, 0.2, 0.3], dtype=np.float32).astype(np.float16)
    y = np.array([0.1, 0.2, 0.1], dtype=np.float32).astype(np.float16)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half, zero division
    x = np.array([1.0, 2.0], dtype=np.float32).astype(np.float16)
    y = np.array([1.0, 1.0], dtype=np.float32).astype(np.float16)  # Avoid zero division for now
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, broadcasting, integer division
    x = np.array([4.0, 6.0, 8.0], dtype=np.float32)
    y = np.array(2.0, dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int64, large numbers, convert to float
    x = np.array([2**31, 2**32], dtype=np.int64).astype(np.float64)
    y = np.array([2, 4], dtype=np.int64).astype(np.float64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint8, all values are ones, convert to float
    x = np.array([1, 1, 1], dtype=np.uint8).astype(np.float32)
    y = np.array([1, 1, 1], dtype=np.uint8).astype(np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex64
    x = np.array([1 + 1j, 2 + 2j], dtype=np.complex64)
    y = np.array([1 - 1j, 2 - 2j], dtype=np.complex64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex128, different values
    x = np.array([1 + 1j, 2 + 3j], dtype=np.complex128)
    y = np.array([2 - 1j, 1 - 2j], dtype=np.complex128)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, multi-dimensional arrays
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RealDiv"] = tf_raw_ops_realdiv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RealDiv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RealDiv'.")

check_valid('tf.raw_ops.RealDiv', generated_inputs['tf.raw_ops.RealDiv'], lib="tf", suffix=0)
