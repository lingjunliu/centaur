
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_bitwise_or_inputs():
    list_of_inputs = []

    # Input 1: Simple case with positive integers
    x1 = tf.constant(np.array([1, 2, 3, 4]))
    x2 = tf.constant(np.array([4, 3, 2, 1]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: With negative integers
    x1 = tf.constant(np.array([-1, -2, -3, -4]))
    x2 = tf.constant(np.array([4, 3, -2, -1]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: With different data types
    x1 = tf.constant(np.array([1, 2, 3, 4], dtype=np.int32))
    x2 = tf.constant(np.array([4, 3, 2, 1], dtype=np.int32))


    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: With zeros
    x1 = tf.constant(np.array([0, 0, 0, 0]))
    x2 = tf.constant(np.array([1, 2, 3, 4]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional arrays
    x1 = tf.constant(np.array([[1, 2], [3, 4]]))
    x2 = tf.constant(np.array([[4, 3], [2, 1]]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different shapes, but broadcastable
    x1 = tf.constant(np.array([1, 2, 3]))
    x2 = tf.constant(np.array([[4], [3], [2]]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger integers
    x1 = tf.constant(np.array([255, 65535]))
    x2 = tf.constant(np.array([128, 32768]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Booleans (treated as 0 and 1)
    x1 = tf.constant(np.array([True, False, True]))
    x2 = tf.constant(np.array([False, True, False]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Reshape bool to int32
    x1 = tf.cast(tf.constant(np.array([True, False, True])), tf.int32)
    x2 = tf.cast(tf.constant(np.array([False, True, False])), tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.bitwise_or"] = tf_experimental_numpy_bitwise_or_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.bitwise_or' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.bitwise_or'.")

check_valid('tf.experimental.numpy.bitwise_or', generated_inputs['tf.experimental.numpy.bitwise_or'], lib="tf", suffix=0)
