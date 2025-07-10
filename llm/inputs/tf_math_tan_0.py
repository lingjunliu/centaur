
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_tan_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "name": "tan_float32_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D array
    x = np.array([[-np.pi/4, 0], [np.pi/4, np.pi/2]], dtype=np.float64)
    input_dict = {"x": x, "name": "tan_float64_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: complex64, 1D array
    x = np.array([1 + 1j, 2 - 2j, 3 + 0j], dtype=np.complex64)
    input_dict = {"x": x, "name": "tan_complex64_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex128, scalar
    x = np.complex128(0.5 - 0.5j)
    input_dict = {"x": x, "name": "tan_complex128_scalar"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: bfloat16, 2D array, negative values
    x = np.array([[-0.5, -1.0], [-1.5, -2.0]], dtype=np.float16)
    input_dict = {"x": x, "name": "tan_bfloat16_2d_neg"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: half, scalar
    x = np.float16(1.5)
    input_dict = {"x":  x, "name": "tan_half_scalar"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, 3D array
    x = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict = {"x": x, "name": "tan_float32_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: float64, large values
    x = np.array([1000.0, -1000.0], dtype=np.float64)
    input_dict = {"x": x, "name": "tan_float64_large"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex64, 2D array
    x = np.array([[1+1j, 2-1j], [0+2j, -1-1j]], dtype=np.complex64)
    input_dict = {"x": x, "name": "tan_complex64_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: bfloat16, larger array
    x = np.random.rand(5, 5).astype(np.float16)
    input_dict = {"x": x, "name": "tan_bfloat16_larger"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_math_tan_inputs()
generated_inputs["tf.math.tan"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.tan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.tan'.")

check_valid('tf.math.tan', generated_inputs['tf.math.tan'], lib="tf", suffix=0)
