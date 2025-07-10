
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_isreal_inputs():
    list_of_inputs = []

    # Input 1: Real tensor
    x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Complex tensor
    x = tf.constant([1.0 + 1j, 2.0 + 2j, 3.0 + 3j], dtype=tf.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Purely real tensor (but complex type)
    x = tf.constant([1.0 + 0j, 2.0 + 0j, 3.0 + 0j], dtype=tf.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-dimensional real tensor
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional complex tensor
    x = tf.constant([[1.0 + 1j, 2.0 + 2j], [3.0 + 3j, 4.0 + 4j]], dtype=tf.complex128)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Scalar real tensor
    x = tf.constant(5.0, dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Scalar complex tensor
    x = tf.constant(5.0 + 2j, dtype=tf.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Real tensor with negative values
    x = tf.constant([-1.0, -2.0, -3.0], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex tensor with negative real and imaginary parts
    x = tf.constant([-1.0 - 1j, -2.0 - 2j, -3.0 - 3j], dtype=tf.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Higher dimensional tensor
    x = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Convert tensors to numpy arrays
    for input_dict in list_of_inputs:
        input_dict['x'] = input_dict['x'].numpy()

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.isreal"] = tf_experimental_numpy_isreal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.isreal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.isreal'.")

check_valid('tf.experimental.numpy.isreal', generated_inputs['tf.experimental.numpy.isreal'], lib="tf", suffix=0)
