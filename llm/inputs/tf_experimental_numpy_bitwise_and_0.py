
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_bitwise_and_inputs():
    list_of_inputs = []

    # Input 1: Basic integers
    x1 = tf.constant(np.array([1, 2, 3, 4, 5], dtype=np.int32))
    x2 = tf.constant(np.array([5, 4, 3, 2, 1], dtype=np.int32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different integer types
    x1 = tf.constant(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    x2 = tf.constant(np.array([5, 4, 3, 2, 1], dtype=np.int64))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Boolean arrays
    x1 = tf.constant(np.array([True, False, True, False], dtype=np.bool_))
    x2 = tf.constant(np.array([False, True, False, True], dtype=np.bool_))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays
    x1 = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int32))
    x2 = tf.constant(np.array([[4, 3], [2, 1]], dtype=np.int32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D arrays
    x1 = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32))
    x2 = tf.constant(np.array([[[8, 7], [6, 5]], [[4, 3], [2, 1]]], dtype=np.int32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative integers
    x1 = tf.constant(np.array([-1, -2, -3, -4, -5], dtype=np.int32))
    x2 = tf.constant(np.array([5, 4, 3, 2, 1], dtype=np.int32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting
    x1 = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    x2 = tf.constant(np.array(1, dtype=np.int32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger integers
    x1 = tf.constant(np.array([2**10, 2**10, 2**10], dtype=np.int32))
    x2 = tf.constant(np.array([2**10, 2**10, 2**10], dtype=np.int32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different shapes, but broadcastable
    x1 = tf.constant(np.array([[1, 2, 3]], dtype=np.int32))
    x2 = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: All zeros
    x1 = tf.constant(np.array([0, 0, 0], dtype=np.int32))
    x2 = tf.constant(np.array([0, 0, 0], dtype=np.int32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.bitwise_and"] = tf_experimental_numpy_bitwise_and_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.bitwise_and' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.bitwise_and'.")

check_valid('tf.experimental.numpy.bitwise_and', generated_inputs['tf.experimental.numpy.bitwise_and'], lib="tf", suffix=0)
