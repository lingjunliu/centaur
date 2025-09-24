
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_squared_difference_inputs():
    list_of_inputs = []

    # Input 1: float32, basic case
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, different values
    x = np.array([-1.5, 2.5, -3.5], dtype=np.float64)
    y = np.array([0.5, -1.5, 2.5], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": "squared_diff_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32, positive and negative integers
    x = np.array([-1, 2, -3, 4], dtype=np.int32)
    y = np.array([5, -6, 7, -8], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64, larger numbers
    x = np.array([1000000000, 2000000000], dtype=np.int64)
    y = np.array([3000000000, 4000000000], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64
    x = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
    y = np.array([4 + 4j, 5 + 5j, 6 + 6j], dtype=np.complex64)
    input_dict = {"x": x, "y": y, "name": "complex_diff"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128
    x = np.array([-1.5 + 0.5j, 2.5 - 1.5j], dtype=np.complex128)
    y = np.array([0.5 - 2.5j, -1.5 + 3.5j], dtype=np.complex128)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: half (float16), some values
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float16)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: multi-dimensional float32
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: multi-dimensional int64
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    y = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: float32, single element arrays
    x = np.array([1.0], dtype=np.float32)
    y = np.array([2.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Complex64, array with a single element
    x = np.array([1 + 2j], dtype=np.complex64)
    y = np.array([3 + 4j], dtype=np.complex64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SquaredDifference"] = tf_raw_ops_squared_difference_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SquaredDifference' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SquaredDifference'.")

check_valid('tf.raw_ops.SquaredDifference', generated_inputs['tf.raw_ops.SquaredDifference'], lib="tf", suffix=0)
