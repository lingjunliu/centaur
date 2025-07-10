
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_divide_inputs():
    list_of_inputs = []

    # Input 1: Basic division with integers
    x1 = tf.constant(np.array([10, 20, 30]), dtype=tf.int32)
    x2 = tf.constant(np.array([2, 5, 10]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Division with floats
    x1 = tf.constant(np.array([1.0, 2.5, 3.7]), dtype=tf.float32)
    x2 = tf.constant(np.array([0.5, 1.0, 2.0]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Division with different shapes (broadcasting)
    x1 = tf.constant(np.array([[1, 2, 3], [4, 5, 6]]), dtype=tf.int32)
    x2 = tf.constant(np.array([1, 2, 3]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Division with negative values
    x1 = tf.constant(np.array([-10, 20, -30]), dtype=tf.int32)
    x2 = tf.constant(np.array([2, -5, 10]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Division with zeros in denominator (should handle gracefully)
    x1 = tf.constant(np.array([1, 2, 3]), dtype=tf.float32)
    x2 = tf.constant(np.array([0, 1, 2]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Multi-dimensional arrays
    x1 = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), dtype=tf.float32)
    x2 = tf.constant(np.array([[[0.5, 1], [1, 0.5]], [[1, 2], [2, 1]]]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large numbers
    x1 = tf.constant(np.array([1e9, 2e9]), dtype=tf.float32)
    x2 = tf.constant(np.array([10, 20]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different data types
    x1 = tf.constant(np.array([1, 2, 3]), dtype=tf.float64)
    x2 = tf.constant(np.array([0.5, 1, 1.5]), dtype=tf.float64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Division by one
    x1 = tf.constant(np.array([5, 10, 15]), dtype=tf.int32)
    x2 = tf.constant(np.array([1, 1, 1]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Broadcasting with scalar
    x1 = tf.constant(np.array([1, 2, 3, 4]), dtype=tf.float32)
    x2 = tf.constant(2.0, dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.divide"] = tf_experimental_numpy_divide_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.divide' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.divide'.")

check_valid('tf.experimental.numpy.divide', generated_inputs['tf.experimental.numpy.divide'], lib="tf", suffix=0)
