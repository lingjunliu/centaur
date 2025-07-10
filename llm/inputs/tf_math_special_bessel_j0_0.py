
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_special_bessel_j0_inputs():
    list_of_inputs = []

    # Input 1
    x = tf.constant(np.array([0.5, 1.0, 2.0]), dtype=tf.float32).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = tf.constant(np.array([-0.5, -1.0, -2.0]), dtype=tf.float32).numpy()
    name = "bessel_j0_negative"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = tf.constant(np.array([0.0, 1.5, 3.0, 4.5]), dtype=tf.float64).numpy()
    name = "bessel_j0_float64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = tf.constant(np.array([[0.5, 1.0], [1.5, 2.0]]), dtype=tf.float32).numpy()
    name = "bessel_j0_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = tf.constant(np.array([[[0.5, 1.0], [1.5, 2.0]], [[2.5, 3.0], [3.5, 4.0]]]), dtype=tf.float32).numpy()
    name = "bessel_j0_3d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = tf.constant(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), dtype=tf.float32).numpy()
    name = "bessel_j0_many_elements"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = tf.constant(np.array([10.0, 20.0, 30.0]), dtype=tf.float32).numpy()
    name = "bessel_j0_large_values"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = tf.constant(np.array([-10.0, -20.0, -30.0]), dtype=tf.float32).numpy()
    name = "bessel_j0_large_negative_values"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = tf.constant(np.array([0.001, 0.002, 0.003]), dtype=tf.float32).numpy()
    name = "bessel_j0_small_values"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = tf.constant(np.array([np.pi, np.e, np.sqrt(2)]), dtype=tf.float32).numpy()
    name = "bessel_j0_constants"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.special.bessel_j0"] = tf_math_special_bessel_j0_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.special.bessel_j0' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.special.bessel_j0'.")

check_valid('tf.math.special.bessel_j0', generated_inputs['tf.math.special.bessel_j0'], lib="tf", suffix=0)
