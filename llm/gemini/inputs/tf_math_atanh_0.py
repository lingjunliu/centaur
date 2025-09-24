
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_atanh_inputs():
    list_of_inputs = []

    # Input 1: float32, scalar
    x = np.float32(0.5)
    name = "atanh_scalar"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, 1D array
    x = np.array([-0.9, -0.5, 0, 0.5, 0.9], dtype=np.float32)
    name = "atanh_1d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, 2D array
    x = np.array([[-0.8, -0.2], [0.3, 0.7]], dtype=np.float32)
    name = "atanh_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, scalar
    x = np.float64(0.25)
    name = "atanh_scalar_double"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64, 1D array
    x = np.array([-0.95, -0.1, 0.1, 0.95], dtype=np.float64)
    name = "atanh_1d_double"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, 2D array
    x = np.array([[-0.6, -0.4], [0.15, 0.85]], dtype=np.float64)
    name = "atanh_2d_double"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: half, scalar
    x = np.float16(0.4)
    name = "atanh_scalar_half"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: half, 1D array
    x = np.array([-0.1, 0.0, 0.1], dtype=np.float16)
    name = "atanh_1d_half"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex64, scalar
    x = np.complex64(0.5 + 0.1j)
    name = "atanh_scalar_complex64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex128, 1D array
    x = np.array([0.2 + 0.1j, 0.4 - 0.2j], dtype=np.complex128)
    name = "atanh_1d_complex128"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.atanh"] = tf_math_atanh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.atanh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.atanh'.")

check_valid('tf.math.atanh', generated_inputs['tf.math.atanh'], lib="tf", suffix=0)
