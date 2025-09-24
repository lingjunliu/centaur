
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_bessel_i0_inputs():
    list_of_inputs = []

    # Input 1: Basic test with float32
    x = tf.constant(np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic test with float64
    x = tf.constant(np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)).numpy()
    name = "bessel_i0_float64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: With larger values
    x = tf.constant(np.array([-5.0, -2.5, 0.0, 2.5, 5.0], dtype=np.float32)).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: With a range of positive values
    x = tf.constant(np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=np.float32)).numpy()
    name = "positive_range"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: With a range of negative values
    x = tf.constant(np.array([-0.1, -0.5, -1.0, -2.0, -5.0, -10.0], dtype=np.float64)).numpy()
    name = "negative_range"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6: Zero input
    x = tf.constant(np.array([0.0], dtype=np.float32)).numpy()
    name = "zero"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D Tensor
    x = tf.constant(np.array([[-1.0, 0.0, 1.0], [-2.0, 0.5, 2.0]], dtype=np.float32)).numpy()
    name = "2d_tensor"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D Tensor
    x = tf.constant(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)).numpy()
    name = "3d_tensor"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large positive values
    x = tf.constant(np.array([100.0, 200.0, 300.0], dtype=np.float32)).numpy()
    name = "large_positive"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large negative values
    x = tf.constant(np.array([-100.0, -200.0, -300.0], dtype=np.float64)).numpy()
    name = "large_negative"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.bessel_i0"] = tf_math_bessel_i0_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.bessel_i0' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.bessel_i0'.")

check_valid('tf.math.bessel_i0', generated_inputs['tf.math.bessel_i0'], lib="tf", suffix=0)
