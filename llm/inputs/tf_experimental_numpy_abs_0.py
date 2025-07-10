
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_abs_inputs():
    list_of_inputs = []

    # Input 1: Scalar integer
    x = tf.constant(-5, dtype=tf.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D integer tensor
    x = tf.constant([-1, 0, 1, -2, 2], dtype=tf.int64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D integer tensor
    x = tf.constant([[-1, 2], [-3, 4]], dtype=tf.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar float
    x = tf.constant(-3.14, dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float tensor
    x = tf.constant([-1.5, 0.0, 1.5, -2.5, 2.5], dtype=tf.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float tensor
    x = tf.constant([[-1.0, 2.0], [-3.0, 4.0]], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex tensor
    x = tf.constant([-1 - 1j, 1 + 1j, 2 - 2j], dtype=tf.complex128)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Higher dimension tensor (3D) with integers
    x = tf.constant([[[1, -2], [3, -4]], [[-5, 6], [-7, 8]]], dtype=tf.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Higher dimension tensor (3D) with floats
    x = tf.constant([[[1.1, -2.2], [3.3, -4.4]], [[-5.5, 6.6], [-7.7, 8.8]]], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Tensor with only negative values
    x = tf.constant([-1, -2, -3, -4], dtype=tf.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_experimental_numpy_abs_inputs()
processed_inputs = []
for input_dict in inputs:
    new_input_dict = {}
    for key, value in input_dict.items():
        new_input_dict[key] = value.numpy() if isinstance(value, tf.Tensor) else value
    processed_inputs.append(new_input_dict)

generated_inputs["tf.experimental.numpy.abs"] = processed_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.abs' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.abs'.")

check_valid('tf.experimental.numpy.abs', generated_inputs['tf.experimental.numpy.abs'], lib="tf", suffix=0)
