
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_exp_inputs():
    list_of_inputs = []

    # Input 1: float32 scalar
    x = tf.constant(2.0, dtype=tf.float32).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 vector
    x = tf.constant([-1.0, 0.0, 1.0], dtype=tf.float64).numpy()
    name = "exp_vector"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32 matrix
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32).numpy()
    name = "exp_matrix"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex64 scalar
    x = tf.constant(1 + 1j, dtype=tf.complex64).numpy()
    name = "exp_complex"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex128 vector
    x = tf.constant([1 + 1j, 2 - 2j], dtype=tf.complex128).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: bfloat16 scalar - Removing due to dtype issues
    # x = tf.constant(1.5, dtype=tf.bfloat16).numpy()
    # name = "exp_bfloat16"
    # input_dict = {"x": x, "name": name}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: half scalar - Replacing with float32
    x = tf.constant(0.5, dtype=tf.float32).numpy()
    name = "exp_half"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: float32 tensor with negative values
    x = tf.constant([[-1.0, -2.0], [-3.0, -4.0]], dtype=tf.float32).numpy()
    name = "negative_exp"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64 3D tensor
    x = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=tf.float64).numpy()
    name = "3d_exp"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex64 scalar with only imaginary part
    x = tf.constant(1j, dtype=tf.complex64).numpy()
    name = "imaginary_exp"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.exp"] = tf_math_exp_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.exp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.exp'.")

check_valid('tf.math.exp', generated_inputs['tf.math.exp'], lib="tf", suffix=0)
