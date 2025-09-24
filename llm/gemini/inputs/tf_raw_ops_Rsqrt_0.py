
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Rsqrt_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D
    x = np.array([1.0, 4.0, 9.0], dtype=np.float32)
    input_dict = {"x": x, "name": "sqrt_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict = {"x": x, "name": "sqrt_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: complex64, 1D
    x = np.array([1.0 + 1j, 2.0 + 2j, 3.0 + 3j], dtype=np.complex64)
    input_dict = {"x": x, "name": "sqrt_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex128, 2D
    x = np.array([[1.0 + 0j, 0 + 1.0j], [-1.0 + 0j, 0 - 1.0j]], dtype=np.complex128)
    input_dict = {"x": x, "name": "sqrt_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: bfloat16, 1D.  Needs to be cast to numpy.
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16).astype(np.float32)
    input_dict = {"x": x, "name": "sqrt_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: half, 2D
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    input_dict = {"x": x, "name": "sqrt_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, 3D
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"x": x, "name": "sqrt_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: float64, 1D with large values
    x = np.array([100.0, 400.0, 900.0], dtype=np.float64)
    input_dict = {"x": x, "name": "sqrt_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: complex64, 1D with zero imaginary part
    x = np.array([1.0 + 0j, 4.0 + 0j, 9.0 + 0j], dtype=np.complex64)
    input_dict = {"x": x, "name": "sqrt_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: bfloat16, 1D with various values
    x = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float16).astype(np.float32)
    input_dict = {"x": x, "name": "sqrt_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Rsqrt"] = tf_raw_ops_Rsqrt_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Rsqrt' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Rsqrt'.")

check_valid('tf.raw_ops.Rsqrt', generated_inputs['tf.raw_ops.Rsqrt'], lib="tf", suffix=0)
