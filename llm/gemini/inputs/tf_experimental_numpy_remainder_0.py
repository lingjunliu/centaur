
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_remainder_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive integers
    x1 = np.array([10, 17, 25], dtype=np.int32)
    x2 = np.array([3, 5, 7], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative integers
    x1 = np.array([-10, -17, -25], dtype=np.int32)
    x2 = np.array([3, -5, 7], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Floating-point numbers
    x1 = np.array([10.5, 17.2, 25.8], dtype=np.float32)
    x2 = np.array([3.0, 5.5, 7.1], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Mixed positive and negative floats
    x1 = np.array([-10.5, 17.2, -25.8], dtype=np.float32)
    x2 = np.array([3.0, -5.5, 7.1], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D arrays
    x1 = np.array([[10, 17], [25, 30]], dtype=np.int32)
    x2 = np.array([[3, 5], [7, 9]], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D arrays with floats
    x1 = np.array([[10.5, 17.2], [25.8, 30.1]], dtype=np.float32)
    x2 = np.array([[3.0, 5.5], [7.1, 9.2]], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different dtypes
    x1 = np.array([10, 17, 25], dtype=np.int64)
    x2 = np.array([3, 5, 7], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Broadcasting
    x1 = np.array([10, 17, 25], dtype=np.int32)
    x2 = np.array(3, dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large numbers
    x1 = np.array([1000000000, 1700000000, 2500000000], dtype=np.int64)
    x2 = np.array([3, 5, 7], dtype=np.int64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.remainder"] = tf_experimental_numpy_remainder_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.remainder' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.remainder'.")

check_valid('tf.experimental.numpy.remainder', generated_inputs['tf.experimental.numpy.remainder'], lib="tf", suffix=0)
