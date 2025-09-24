
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_special_fresnel_sin_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array, positive values
    x = tf.constant(np.array([0.1, 0.5, 1.0], dtype=np.float32)).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, 1D array, negative values
    x = tf.constant(np.array([-0.1, -0.5, -1.0], dtype=np.float32)).numpy()
    name = "negative_values"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, scalar value
    x = tf.constant(np.float64(2.0)).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, 1D array, mixed values
    x = tf.constant(np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)).numpy()
    name = "mixed_values"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, 2D array
    x = tf.constant(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, 2D array, negative values
    x = tf.constant(np.array([[-0.1, -0.2], [-0.3, -0.4]], dtype=np.float64)).numpy()
    name = "2d_negative"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, 3D array
    x = tf.constant(np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, scalar, zero
    x = tf.constant(np.float64(0.0)).numpy()
    name = "zero"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, large positive value
    x = tf.constant(np.float32(10.0)).numpy()
    name = "large_positive"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64, large negative value
    x = tf.constant(np.float64(-10.0)).numpy()
    name = "large_negative"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.special.fresnel_sin"] = tf_math_special_fresnel_sin_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.special.fresnel_sin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.special.fresnel_sin'.")

check_valid('tf.math.special.fresnel_sin', generated_inputs['tf.math.special.fresnel_sin'], lib="tf", suffix=0)
