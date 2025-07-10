
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_greater_inputs():
    list_of_inputs = []

    # Input 1: Basic test with positive integers
    x1 = tf.constant(np.array([1, 2, 3]))
    x2 = tf.constant(np.array([0, 2, 4]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Test with negative integers
    x1 = tf.constant(np.array([-1, -2, -3]))
    x2 = tf.constant(np.array([-2, -2, -4]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Test with mixed positive and negative integers
    x1 = tf.constant(np.array([-1, 2, -3]))
    x2 = tf.constant(np.array([0, -2, 4]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Test with floating point numbers
    x1 = tf.constant(np.array([1.0, 2.5, 3.2]))
    x2 = tf.constant(np.array([0.5, 2.5, 4.0]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Test with 2D arrays
    x1 = tf.constant(np.array([[1, 2], [3, 4]]))
    x2 = tf.constant(np.array([[0, 3], [2, 5]]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Test with 3D arrays
    x1 = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    x2 = tf.constant(np.array([[[0, 3], [2, 5]], [[4, 7], [6, 9]]]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: Test with different shapes but broadcastable
    x1 = tf.constant(np.array([1, 2, 3]))
    x2 = tf.constant(np.array(2))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Test with all elements being the same
    x1 = tf.constant(np.array([5, 5, 5]))
    x2 = tf.constant(np.array([4, 4, 4]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.greater"] = tf_experimental_numpy_greater_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.greater' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.greater'.")

check_valid('tf.experimental.numpy.greater', generated_inputs['tf.experimental.numpy.greater'], lib="tf", suffix=0)
