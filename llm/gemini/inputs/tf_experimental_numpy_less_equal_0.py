
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_less_equal_inputs():
    list_of_inputs = []

    # Input 1: Basic integer tensors
    x1 = tf.constant(np.array([1, 2, 3]), dtype=tf.int32)
    x2 = tf.constant(np.array([2, 2, 1]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic float tensors
    x1 = tf.constant(np.array([1.0, 2.0, 3.0]), dtype=tf.float32)
    x2 = tf.constant(np.array([2.0, 2.0, 1.0]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values
    x1 = tf.constant(np.array([-1, -2, -3]), dtype=tf.int32)
    x2 = tf.constant(np.array([0, -2, -1]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-dimensional tensors
    x1 = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.int32)
    x2 = tf.constant(np.array([[2, 1], [4, 3]]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different shapes (broadcastable)
    x1 = tf.constant(np.array([1, 2, 3]), dtype=tf.int32)
    x2 = tf.constant(np.array(2), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Same values
    x1 = tf.constant(np.array([1, 1, 1]), dtype=tf.int32)
    x2 = tf.constant(np.array([1, 1, 1]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger values
    x1 = tf.constant(np.array([1000, 2000, 3000]), dtype=tf.int32)
    x2 = tf.constant(np.array([2000, 1000, 3000]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Broadcasting with multi-dimensional array
    x1 = tf.constant(np.array([[1, 2, 3], [4, 5, 6]]), dtype=tf.int32)
    x2 = tf.constant(np.array([2, 4, 5]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger multi-dimensional tensors with negative values
    x1 = tf.constant(np.array([[-1, 2, -3], [4, -5, 6], [-7, 8, -9]]), dtype=tf.int32)
    x2 = tf.constant(np.array([[2, -1, 4], [-3, 6, -5], [8, -9, 7]]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float64
    x1 = tf.constant(np.array([1.0, 2.0, 3.0]), dtype=tf.float64)
    x2 = tf.constant(np.array([2.0, 2.0, 1.0]), dtype=tf.float64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.less_equal"] = tf_experimental_numpy_less_equal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.less_equal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.less_equal'.")

check_valid('tf.experimental.numpy.less_equal', generated_inputs['tf.experimental.numpy.less_equal'], lib="tf", suffix=0)
