
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_sinh_inputs():
    list_of_inputs = []

    # Input 1: Scalar
    x = tf.constant(0.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive scalar
    x = tf.constant(1.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative scalar
    x = tf.constant(-1.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array
    x = tf.constant(np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array
    x = tf.constant(np.array([[0.0, 1.0], [-1.0, 2.0], [-2.0, 0.5]], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array
    x = tf.constant(np.array([[[0.0, 1.0], [-1.0, 2.0]], [[-2.0, 0.5], [1.5, -0.75]]], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large positive value
    x = tf.constant(10.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large negative value
    x = tf.constant(-10.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Array with different data types
    x = tf.cast(tf.constant(np.array([0, 1, -1, 2, -2], dtype=np.int32)), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array with complex numbers - Removing complex numbers as min/max op does not support complex types
    # x = tf.constant(np.array([1+1j, 2-2j, -1+0j, 0-1j], dtype=np.complex64))
    # input_dict = {"x": x}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.sinh"] = tf_experimental_numpy_sinh_inputs()

tf.experimental.numpy.experimental_enable_numpy_behavior()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.sinh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.sinh'.")

check_valid('tf.experimental.numpy.sinh', generated_inputs['tf.experimental.numpy.sinh'], lib="tf", suffix=0)
