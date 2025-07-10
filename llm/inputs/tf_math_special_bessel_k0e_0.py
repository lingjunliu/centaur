
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_special_bessel_k0e_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    x = tf.constant([0.5, 1.0, 2.0], dtype=tf.float32).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different float type (float64)
    x = tf.constant([0.5, 1.0, 2.0], dtype=tf.float64).numpy()
    name = "bessel_k0e_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: A single value
    x = tf.constant(1.5, dtype=tf.float32).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative value (valid, as per documentation)
    x = tf.constant([-0.5, -1.0, -2.0], dtype=tf.float32).numpy()
    name = "bessel_k0e_3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Array with a larger range of values
    x = tf.constant([0.1, 1.0, 5.0, 10.0], dtype=tf.float32).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multi-dimensional tensor (2D)
    x = tf.constant([[0.5, 1.0], [2.0, 3.0]], dtype=tf.float32).numpy()
    name = "bessel_k0e_4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with zeros
    x = tf.constant([0.0, 0.5, 1.0], dtype=tf.float32).numpy()
    name = "bessel_k0e_5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: More complex values
    x = tf.constant([0.25, 0.75, 1.25, 1.75], dtype=tf.float32).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: large values
    x = tf.constant([100.0, 200.0, 300.0], dtype=tf.float32).numpy()
    name = "bessel_k0e_6"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10: Multi-dimensional tensor (3D)
    x = tf.constant([[[0.5, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]], dtype=tf.float32).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.special.bessel_k0e"] = tf_math_special_bessel_k0e_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.special.bessel_k0e' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.special.bessel_k0e'.")

check_valid('tf.math.special.bessel_k0e', generated_inputs['tf.math.special.bessel_k0e'], lib="tf", suffix=0)
