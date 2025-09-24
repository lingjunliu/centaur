
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_gcd_inputs():
    list_of_inputs = []

    # Input 1: Basic positive integers
    x1 = tf.constant(np.array([12, 24, 36]), dtype=tf.int32)
    x2 = tf.constant(np.array([18, 36, 48]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Mixed positive integers and zeros
    x1 = tf.constant(np.array([0, 15, 20]), dtype=tf.int32)
    x2 = tf.constant(np.array([5, 0, 30]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative integers (should be treated as absolute values)
    x1 = tf.constant(np.array([-12, -24, -36]), dtype=tf.int32)
    x2 = tf.constant(np.array([-18, -36, -48]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Mixed positive and negative integers
    x1 = tf.constant(np.array([12, -24, 36]), dtype=tf.int32)
    x2 = tf.constant(np.array([-18, 36, -48]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger integers
    x1 = tf.constant(np.array([12345, 67890]), dtype=tf.int32)
    x2 = tf.constant(np.array([98765, 43210]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multi-dimensional arrays
    x1 = tf.constant(np.array([[12, 24], [36, 48]]), dtype=tf.int32)
    x2 = tf.constant(np.array([[18, 36], [48, 60]]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different data types (int64)
    x1 = tf.constant(np.array([12, 24, 36]), dtype=tf.int64)
    x2 = tf.constant(np.array([18, 36, 48]), dtype=tf.int64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Ones and other numbers
    x1 = tf.constant(np.array([1, 5, 7]), dtype=tf.int32)
    x2 = tf.constant(np.array([3, 1, 11]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.gcd"] = tf_experimental_numpy_gcd_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.gcd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.gcd'.")

check_valid('tf.experimental.numpy.gcd', generated_inputs['tf.experimental.numpy.gcd'], lib="tf", suffix=0)
