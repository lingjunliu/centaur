
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_reciprocal_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    name = "reciprocal_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D array
    x = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float64)
    name = "reciprocal_2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: complex64, 1D array
    x = np.array([1 + 1j, 2 - 2j, 3 + 0j], dtype=np.complex64)
    name = "reciprocal_3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex128, 2D array
    x = np.array([[1 + 1j, 2 - 2j], [3 + 0j, 0 + 4j]], dtype=np.complex128)
    name = "reciprocal_4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, 3D array
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    name = "reciprocal_5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: bfloat16, 1D array
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).astype(np.float16) #cast to float16
    name = "reciprocal_8"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: half, 2D array
    x = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32).astype(np.float16)
    name = "reciprocal_9"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: float32 with zero
    x = np.array([0.01, 0.1, 1.0, 0.0], dtype=np.float32)
    name = "reciprocal_10"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: float64, scalar value
    x = np.array(2.5, dtype=np.float64)
    name = "reciprocal_11"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: float32, empty array
    x = np.array([], dtype=np.float32)
    name = "reciprocal_12"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.reciprocal"] = tf_math_reciprocal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.reciprocal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.reciprocal'.")

check_valid('tf.math.reciprocal', generated_inputs['tf.math.reciprocal'], lib="tf", suffix=0)
