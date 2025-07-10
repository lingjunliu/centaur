
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_cosh_inputs():
    list_of_inputs = []

    # Input 1: float32, single element
    x = np.array(1.0, dtype=np.float32)
    name = None
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, 1D array, negative values
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    name = "cosh_1d"
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, 2D array
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    name = None
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex64, single element
    x = np.complex64(1 + 1j)
    name = None
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex128, 1D array
    x = np.array([1 + 1j, 2 - 2j, 3 + 0j], dtype=np.complex128)
    name = "cosh_complex"
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32, large values
    x = np.array([100.0, -100.0], dtype=np.float32)
    name = "cosh_large"
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float16, single element, negative
    x = np.array(-2.5, dtype=np.float16)
    name = None
    input_dict = {"x": tf.constant(x, dtype=tf.float16), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, 3D array, negative and positive
    x = np.array([[[1.0, -2.0], [3.0, -4.0]], [[-5.0, 6.0], [-7.0, 8.0]]], dtype=np.float64)
    name = "cosh_3d"
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32
    x = np.array([0.0], dtype=np.float32)
    name = None
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32
    x = np.array([-5.0, 5.0], dtype=np.float32)
    name = None
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.cosh"] = tf_math_cosh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.cosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.cosh'.")

check_valid('tf.math.cosh', generated_inputs['tf.math.cosh'], lib="tf", suffix=0)
