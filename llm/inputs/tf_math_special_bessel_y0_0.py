
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_special_bessel_y0_inputs():
    list_of_inputs = []

    # Input 1
    x = tf.constant(np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32))
    name = None
    input_dict = {"x": x, "name": name}
    input_dict["x"] = input_dict["x"].numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = tf.constant(np.array([1.5], dtype=np.float32))
    name = "bessel_y0_2"
    input_dict = {"x": x, "name": name}
    input_dict["x"] = input_dict["x"].numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = tf.constant(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64))
    name = None
    input_dict = {"x": x, "name": name}
    input_dict["x"] = input_dict["x"].numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = tf.constant(np.array([1e-8, 1e-7], dtype=np.float32))
    name = "bessel_y0_4"
    input_dict = {"x": x, "name": name}
    input_dict["x"] = input_dict["x"].numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = tf.constant(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32))
    name = None
    input_dict = {"x": x, "name": name}
    input_dict["x"] = input_dict["x"].numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    x = tf.constant(np.array([0.0], dtype=np.float32))
    name = "bessel_y0_6"
    input_dict = {"x": x, "name": name}
    input_dict["x"] = input_dict["x"].numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = tf.constant(np.array([-0.5], dtype=np.float32))
    name = None
    input_dict = {"x": x, "name": name}
    input_dict["x"] = input_dict["x"].numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = tf.constant(np.array([[-0.1, -0.2], [-0.3, -0.4]], dtype=np.float64))
    name = "bessel_y0_8"
    input_dict = {"x": x, "name": name}
    input_dict["x"] = input_dict["x"].numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = tf.constant(np.array([1e-5], dtype=np.float64))
    name = None
    input_dict = {"x": x, "name": name}
    input_dict["x"] = input_dict["x"].numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = tf.constant(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64))
    name = "bessel_y0_10"
    input_dict = {"x": x, "name": name}
    input_dict["x"] = input_dict["x"].numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.special.bessel_y0"] = tf_math_special_bessel_y0_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.special.bessel_y0' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.special.bessel_y0'.")

check_valid('tf.math.special.bessel_y0', generated_inputs['tf.math.special.bessel_y0'], lib="tf", suffix=0)
