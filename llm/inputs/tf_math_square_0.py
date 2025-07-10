
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_square_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    name = "float32_1d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32, 2D array
    x = np.array([[-1, 0, 1], [2, 3, 4]], dtype=np.int32)
    name = "int32_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, 3D array
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    name = "float64_3d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64, scalar
    x = np.array(-5, dtype=np.int64)
    name = "int64_scalar"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64, 1D array
    x = np.array([1+1j, 2-2j, 3+0j], dtype=np.complex64)
    name = "complex64_1d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128, 2D array
    x = np.array([[1+1j, 2-2j], [3+0j, 4-1j]], dtype=np.complex128)
    name = "complex128_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint8, 2D array
    x = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    name = "uint8_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float16, 1D array
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float16)
    name = "float16_1d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int8, 1D array
    x = np.array([-2, -1, 0, 1, 2], dtype=np.int8)
    name = "int8_1d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.square"] = tf_math_square_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.square' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.square'.")

check_valid('tf.math.square', generated_inputs['tf.math.square'], lib="tf", suffix=0)
