
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_atan_inputs():
    list_of_inputs = []

    # Input 1: float32, scalar
    x = np.array(1.0, dtype=np.float32)
    name = "atan_scalar"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 1D array
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    name = "atan_1d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, 2D array
    x = np.array([[0.5, -0.5], [1.5, -1.5]], dtype=np.float32)
    name = "atan_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, 3D array
    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float64)
    name = "atan_3d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, large values
    x = np.array([1000.0, -1000.0], dtype=np.float32)
    name = "atan_large_values"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, small values
    x = np.array([0.0001, -0.0001], dtype=np.float64)
    name = "atan_small_values"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex64, scalar
    x = np.complex64(1 + 1j)
    name = "atan_complex64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex128, scalar
    x = np.complex128(2 - 2j)
    name = "atan_complex128"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: half, scalar
    x = np.array(0.25, dtype=np.float16)
    name = "atan_half"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: bfloat16, scalar
    x = np.array(0.75, dtype=np.float16)
    name = "atan_bfloat16"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.atan"] = tf_math_atan_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.atan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.atan'.")

check_valid('tf.math.atan', generated_inputs['tf.math.atan'], lib="tf", suffix=0)
