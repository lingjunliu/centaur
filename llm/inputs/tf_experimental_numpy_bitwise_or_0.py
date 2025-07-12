
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_bitwise_or_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive integers
    x1 = np.array([1, 2, 3, 4], dtype=np.int32)
    x2 = np.array([4, 3, 2, 1], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Case with negative integers
    x1 = np.array([-1, -2, -3, -4], dtype=np.int32)
    x2 = np.array([4, 3, 2, -1], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Case with zeros
    x1 = np.array([0, 1, 0, 1], dtype=np.int32)
    x2 = np.array([1, 0, 1, 0], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Case with large integers
    x1 = np.array([2**31 - 1, 2**30, 2**20], dtype=np.int64)
    x2 = np.array([2**30, 2**20, 2**10], dtype=np.int64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Case with boolean arrays
    x1 = np.array([True, False, True, False], dtype=np.bool_)
    x2 = np.array([False, True, False, True], dtype=np.bool_)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multi-dimensional array (2D)
    x1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    x2 = np.array([[4, 3], [2, 1]], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multi-dimensional array (3D)
    x1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    x2 = np.array([[[8, 7], [6, 5]], [[4, 3], [2, 1]]], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Mixed positive and negative numbers with different dtypes.
    x1 = np.array([1, -2, 3, -4], dtype=np.int64)
    x2 = np.array([-4, 3, -2, 1], dtype=np.int64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: All same numbers in both arrays.
    x1 = np.array([5, 5, 5, 5], dtype=np.int32)
    x2 = np.array([5, 5, 5, 5], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10: More complex mix of positive, negative and zero values
    x1 = np.array([-5, 0, 5, -10], dtype=np.int32)
    x2 = np.array([10, -5, 0, 5], dtype=np.int32)
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
