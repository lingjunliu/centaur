
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_subtract_inputs():
    list_of_inputs = []

    # Input 1: Basic subtraction with positive integers
    x1 = tf.constant(np.array([5, 10, 15]))
    x2 = tf.constant(np.array([1, 2, 3]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Subtraction with negative integers
    x1 = tf.constant(np.array([-5, 10, -15]))
    x2 = tf.constant(np.array([1, -2, 3]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Subtraction with floating-point numbers
    x1 = tf.constant(np.array([5.5, 10.2, 15.7]))
    x2 = tf.constant(np.array([1.1, 2.2, 3.3]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Subtraction with a scalar
    x1 = tf.constant(np.array([5, 10, 15]))
    x2 = tf.constant(2)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Subtraction with two-dimensional arrays
    x1 = tf.constant(np.array([[1, 2], [3, 4]]))
    x2 = tf.constant(np.array([[5, 6], [7, 8]]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Subtraction with broadcasting
    x1 = tf.constant(np.array([[1, 2, 3], [4, 5, 6]]))
    x2 = tf.constant(np.array([1, 2, 3]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.subtract"] = tf_experimental_numpy_subtract_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.subtract' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.subtract'.")

check_valid('tf.experimental.numpy.subtract', generated_inputs['tf.experimental.numpy.subtract'], lib="tf", suffix=0)
