
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_bitwise_xor_inputs():
    list_of_inputs = []

    # Input 1: Basic integers
    x1 = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    x2 = tf.constant(np.array([4, 5, 6], dtype=np.int32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shapes, but broadcastable
    x1 = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int32))
    x2 = tf.constant(np.array([5, 6], dtype=np.int32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Boolean values (converted to integers)
    x1 = tf.constant(np.array([True, False, True], dtype=np.bool_))
    x2 = tf.constant(np.array([False, True, False], dtype=np.bool_))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative integers
    x1 = tf.constant(np.array([-1, -2, -3], dtype=np.int32))
    x2 = tf.constant(np.array([4, 5, -6], dtype=np.int32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger integers
    x1 = tf.constant(np.array([255, 65535, 2147483647], dtype=np.int32))
    x2 = tf.constant(np.array([16, 256, 1024], dtype=np.int32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Multi-dimensional arrays
    x1 = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32))
    x2 = tf.constant(np.array([[[8, 7], [6, 5]], [[4, 3], [2, 1]]], dtype=np.int32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero values
    x1 = tf.constant(np.array([0, 0, 0], dtype=np.int32))
    x2 = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: All ones
    x1 = tf.constant(np.array([1, 1, 1], dtype=np.int32))
    x2 = tf.constant(np.array([1, 1, 1], dtype=np.int32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Mix of positive and negative, large and small
    x1 = tf.constant(np.array([-2, 1000, -65536], dtype=np.int32))
    x2 = tf.constant(np.array([65536, -1000, 2], dtype=np.int32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different data types (uint8)
    x1 = tf.constant(np.array([1, 2, 3], dtype=np.uint8))
    x2 = tf.constant(np.array([4, 5, 6], dtype=np.uint8))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.bitwise_xor"] = tf_experimental_numpy_bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.bitwise_xor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.bitwise_xor'.")

check_valid('tf.experimental.numpy.bitwise_xor', generated_inputs['tf.experimental.numpy.bitwise_xor'], lib="tf", suffix=0)
