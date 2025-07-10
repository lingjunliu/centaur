
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_sinh_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D array
    x = np.array([[-2.0, -1.0], [1.0, 2.0]], dtype=np.float64)
    input_dict = {"x": tf.constant(x), "name": "sinh_example"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float16, scalar
    x = np.array(-0.25, dtype=np.float16)
    input_dict = {"x": tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half, 3D array
    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float16)
    input_dict = {"x": tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64, 1D array
    x = np.array([1+1j, 2-2j, -1+0j], dtype=np.complex64)
    input_dict = {"x": tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128, scalar
    x = np.array(1j, dtype=np.complex128)
    input_dict = {"x": tf.constant(x), "name": "complex_sinh"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, all zeros
    x = np.zeros((2, 3), dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, large values
    x = np.array([-100, 100], dtype=np.float64)
    input_dict = {"x": tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, mixed positive and negative values
    x = np.array([-5, -2.5, 0, 2.5, 5], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: half, different shape
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16).reshape((2,2))
    input_dict = {"x": tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.sinh"] = tf_math_sinh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.sinh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.sinh'.")

check_valid('tf.math.sinh', generated_inputs['tf.math.sinh'], lib="tf", suffix=0)
