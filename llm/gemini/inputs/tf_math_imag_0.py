
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_imag_inputs():
    list_of_inputs = []

    # Input 1: complex64, simple case
    x = np.array([-2.25 + 4.75j, 3.25 + 5.75j], dtype=np.complex64)
    input_dict = {"input": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: complex128, simple case
    x = np.array([-2.25 + 4.75j, 3.25 + 5.75j], dtype=np.complex128)
    input_dict = {"input": x, "name": "imag_part"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, should return zeros
    x = np.array([-2.25, 3.25], dtype=np.float32)
    input_dict = {"input": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, should return zeros
    x = np.array([-2.25, 3.25], dtype=np.float64)
    input_dict = {"input":  x, "name": "float_imag"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64, multi-dimensional
    x = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    input_dict = {"input": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128, multi-dimensional with negative imaginary parts
    x = np.array([[1-2j, 3-4j], [5-6j, 7-8j]], dtype=np.complex128)
    input_dict = {"input": x, "name": "neg_imag"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, multi-dimensional
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"input": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, multi-dimensional
    x = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float64)
    input_dict = {"input": x, "name": "float_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex64, zero values
    x = np.array([0+0j, 0+0j], dtype=np.complex64)
    input_dict = {"input": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.imag"] = tf_math_imag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.imag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.imag'.")

check_valid('tf.math.imag', generated_inputs['tf.math.imag'], lib="tf", suffix=0)
