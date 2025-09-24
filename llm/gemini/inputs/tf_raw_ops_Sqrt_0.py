
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sqrt_inputs():
    list_of_inputs = []

    # Input 1: Valid, float32, scalar
    x = np.array(4.0, dtype=np.float32)
    input_dict = {"x": x, "name": "sqrt_scalar"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Valid, float64, 1D array
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {"x": x, "name": "sqrt_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Valid, float32, 2D array (replaces bfloat16)
    x = np.array([[1.0, 4.0], [9.0, 16.0]], dtype=np.float32)
    input_dict = {"x": x, "name": "sqrt_2d_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Valid, half (float16), 3D array
    x = np.array([[[1.0, 4.0], [9.0, 16.0]], [[25.0, 36.0], [49.0, 64.0]]], dtype=np.float16)
    input_dict = {"x": x, "name": "sqrt_3d_half"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Valid, complex64, scalar
    x = np.complex64(4 + 3j)
    input_dict = {"x": x, "name": "sqrt_complex64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Valid, complex128, 1D array
    x = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex128)
    input_dict = {"x": x, "name": "sqrt_complex128"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Valid, float32, array with zero and positive values
    x = np.array([0.0, 1.0, 2.25, 4.0], dtype=np.float32)
    input_dict = {"x": x, "name": "sqrt_float32_positive_zero"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Valid, float64, large values
    x = np.array([1e10, 1e20, 1e30], dtype=np.float64)
    input_dict = {"x": x, "name": "sqrt_float64_large"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Valid, complex64, multi-dimensional array
    x = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]], dtype=np.complex64)
    input_dict = {"x": x, "name": "sqrt_complex64_multidim"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Valid, float32, matrix
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"x": x, "name": "sqrt_float32_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Valid, float64, scalar
    x = np.array(9.0, dtype=np.float64)
    input_dict = {"x": x, "name": "sqrt_float64_scalar"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Sqrt"] = tf_raw_ops_sqrt_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Sqrt' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Sqrt'.")

check_valid('tf.raw_ops.Sqrt', generated_inputs['tf.raw_ops.Sqrt'], lib="tf", suffix=0)
