
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Tan_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array
    x = np.array([-1.0, 0.0, 1.0, np.pi/4, np.pi/2], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": "tan_float32_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D array
    x = np.array([[-np.pi/3, -np.pi/6], [np.pi/6, np.pi/3]], dtype=np.float64)
    input_dict = {"x": tf.constant(x), "name": "tan_float64_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float16, scalar
    x = np.array(0.5, dtype=np.float16)
    input_dict = {"x": tf.constant(x, dtype=tf.float16), "name": "tan_bfloat16_scalar"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half, 3D array
    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float16)
    input_dict = {"x": tf.constant(x), "name": "tan_half_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64, 1D array
    x = np.array([1+1j, 2-2j, 3+0j, 0-4j], dtype=np.complex64)
    input_dict = {"x": tf.constant(x), "name": "tan_complex64_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128, scalar
    x = np.array(1j, dtype=np.complex128)
    input_dict = {"x": tf.constant(x), "name": "tan_complex128_scalar"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, small values
    x = np.array([1e-2, -1e-2], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": "tan_float32_large"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, small values
    x = np.array([1e-10, -1e-10], dtype=np.float64)
    input_dict = {"x": tf.constant(x), "name": "tan_float64_small"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float16, zero
    x = np.array(0.0, dtype=np.float16)
    input_dict = {"x": tf.constant(x, dtype=tf.float16), "name": "tan_bfloat16_zero"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: half, negative values
    x = np.array([-0.5, -1.5], dtype=np.float16)
    input_dict = {"x": tf.constant(x), "name": "tan_half_negative"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Tan"] = tf_raw_ops_Tan_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Tan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Tan'.")

check_valid('tf.raw_ops.Tan', generated_inputs['tf.raw_ops.Tan'], lib="tf", suffix=0)
