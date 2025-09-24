
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_sqrt_inputs():
    list_of_inputs = []

    # Input 1: float32, positive values
    x = np.array([[4.0, 9.0], [16.0, 25.0]], dtype=np.float32)
    name = "sqrt_input_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, positive and zero values
    x = np.array([[0.0, 1.0], [4.0, 9.0]], dtype=np.float64)
    name = "sqrt_input_2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float16, positive values
    x = np.array([[1.0, 4.0], [9.0, 16.0]], dtype=np.float16)
    name = "sqrt_input_3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32, negative values (will result in NaN)
    x = np.array([[-1.0, 4.0], [-9.0, 16.0]], dtype=np.float32)
    name = "sqrt_input_5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64
    x = np.array([[1+1j, 4+0j], [0-9j, 16+2j]], dtype=np.complex64)
    name = "sqrt_input_6"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128
    x = np.array([[1+1j, 4+0j], [0-9j, 16+2j]], dtype=np.complex128)
    name = "sqrt_input_7"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, 3D tensor
    x = np.array([[[4.0, 9.0], [16.0, 25.0]], [[1.0, 0.0], [36.0, 49.0]]], dtype=np.float32)
    name = "sqrt_input_8"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, 1D tensor
    x = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float64)
    name = "sqrt_input_9"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, empty array
    x = np.array([], dtype=np.float32)
    name = "sqrt_input_10"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: float32, scalar
    x = np.array(9.0, dtype=np.float32)
    name = "sqrt_input_11"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.sqrt"] = tf_math_sqrt_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.sqrt' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.sqrt'.")

check_valid('tf.math.sqrt', generated_inputs['tf.math.sqrt'], lib="tf", suffix=0)
