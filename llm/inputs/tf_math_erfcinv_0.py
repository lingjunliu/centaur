
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_erfcinv_inputs():
    list_of_inputs = []

    # Input 1: Basic example from the documentation
    x = np.array([0., 0.2, 1., 1.5, 2.], dtype=np.float32)
    name = None
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Double precision
    x = np.array([0., 0.5, 1., 1.8, 2.], dtype=np.float64)
    name = "double_precision"
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single element
    x = np.array(0.7, dtype=np.float32)
    name = "single_element"
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Array with zeros
    x = np.array([0., 0., 0.], dtype=np.float32)
    name = "zeros"
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Array with ones
    x = np.array([1., 1., 1.], dtype=np.float32)
    name = "ones"
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Range of values
    x = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
    name = "range"
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Values close to the boundaries
    x = np.array([0.001, 1.999], dtype=np.float32)
    name = "close_to_boundaries"
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array
    x = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    name = "2d_array"
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array
    x = np.array([[[0.2, 0.4], [0.6, 0.8]], [[1.2, 1.4], [1.6, 1.8]]], dtype=np.float32)
    name = "3d_array"
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed values including 0 and 2
    x = np.array([0., 0.5, 1., 1.5, 2.], dtype=np.float32)
    name = "mixed_values"
    input_dict = {"x": tf.constant(x), "name": name}
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
