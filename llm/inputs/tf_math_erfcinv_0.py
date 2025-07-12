
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_erfcinv_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    x = np.array([0.0, 0.2, 1.0, 1.5, 2.0], dtype=np.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different data type (double)
    x = np.array([0.1, 0.5, 1.2, 1.8], dtype=np.float64)
    name = "erfcinv_double"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional array
    x = np.array([[0.3, 0.7], [1.1, 1.6]], dtype=np.float32)
    name = "erfcinv_multi"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: All zeros
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    name = "erfcinv_zeros"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: All ones
    x = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    name = "erfcinv_ones"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Mixed values
    x = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
    name = "erfcinv_mixed"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Another multi-dimensional array
    x = np.array([[[0.2, 0.4], [0.6, 0.8]], [[1.2, 1.4], [1.6, 1.8]]], dtype=np.float32)
    name = "erfcinv_3d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Values close to the boundaries [0, 2]
    x = np.array([0.0001, 1.9999], dtype=np.float32)
    name = "erfcinv_boundaries"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger array
    x = np.linspace(0.1, 1.9, num=100, dtype=np.float32)
    name = "erfcinv_large"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Name is an empty string
    x = np.array([0.5], dtype=np.float32)
    name = ""
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.erfcinv"] = tf_math_erfcinv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.erfcinv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.erfcinv'.")

check_valid('tf.math.erfcinv', generated_inputs['tf.math.erfcinv'], lib="tf", suffix=0)
