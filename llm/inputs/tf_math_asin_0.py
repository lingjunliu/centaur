
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_asin_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array
    x = np.array([0.0, 0.5, -0.5, 0.9, -0.9], dtype=np.float32)
    name = "asin_float32_1d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D array
    x = np.array([[0.2, 0.4], [-0.6, -0.8]], dtype=np.float64)
    name = "asin_float64_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: half, 1D array
    x = np.array([0.1, -0.3, 0.7], dtype=np.float16)
    name = "asin_half_1d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float16, scalar
    x = np.array(0.9, dtype=np.float16)
    name = "asin_float16_scalar"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64, 1D array
    x = np.array([0.1+0j, -0.2+0j, 0.3+0j], dtype=np.complex64)
    name = "asin_complex64_1d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128, 2D array
    x = np.array([[0.4+0j, -0.5+0j], [0.6+0j, -0.7+0j]], dtype=np.complex128)
    name = "asin_complex128_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, 3D array
    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[-0.1, -0.2], [-0.3, -0.4]]], dtype=np.float32)
    name = "asin_float32_3d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, scalar
    x = np.array(0.6, dtype=np.float64)
    name = "asin_float64_scalar"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, 1D array with values close to 1 and -1
    x = np.array([0.99, -0.99, 0.0], dtype=np.float32)
    name = "asin_float32_close_to_limit"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: half, 2D array with edge cases
    x = np.array([[0.0, 0.9], [-0.9, 0.5]], dtype=np.float16)
    name = "asin_half_2d_edge_cases"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
     # Input 11: float32, with zero
    x = np.array([0.0], dtype=np.float32)
    name = "asin_float32_zero"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.asin"] = tf_math_asin_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.asin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.asin'.")

check_valid('tf.math.asin', generated_inputs['tf.math.asin'], lib="tf", suffix=0)
