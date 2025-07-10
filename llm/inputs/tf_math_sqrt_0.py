
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_sqrt_inputs():
    list_of_inputs = []

    # Input 1: float32, positive values
    x = tf.constant(np.array([[4.0, 9.0], [16.0, 25.0]], dtype=np.float32))
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, positive and zero values
    x = tf.constant(np.array([[0.0, 1.0], [2.25, 4.0]], dtype=np.float64))
    name = "sqrt_op"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: complex64, positive values
    x = tf.constant(np.array([[1.0+0j, 4.0+0j], [9.0+0j, 16.0+0j]], dtype=np.complex64))
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex128, negative values (imaginary result)
    x = tf.constant(np.array([[-1.0+0j, -4.0+0j], [9.0+0j, 16.0+0j]], dtype=np.complex128))
    name = "sqrt_complex"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, multi-dimensional array
    x = tf.constant(np.array([[[1.0, 4.0], [9.0, 16.0]], [[25.0, 36.0], [49.0, 64.0]]], dtype=np.float32))
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, single value
    x = tf.constant(np.array(2.0, dtype=np.float64))
    name = "single_value"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex128, with both real and imaginary parts
    x = tf.constant(np.array([[1.0+1j, 4.0-2j], [-9.0+3j, 16.0-4j]], dtype=np.complex128))
    name = "complex_values"
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
