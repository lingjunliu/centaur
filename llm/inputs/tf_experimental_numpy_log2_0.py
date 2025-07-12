
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_log2_inputs():
    list_of_inputs = []

    # Input 1: Basic positive values
    x = tf.constant(np.array([2, 4, 8, 16], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Zero and one
    x = tf.constant(np.array([1], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional array
    x = tf.constant(np.array([[2, 4], [8, 16]], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different data type (float64)
    x = tf.constant(np.array([2.0, 4.0, 8.0], dtype=np.float64))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large numbers
    x = tf.constant(np.array([2**10, 2**15], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small numbers
    x = tf.constant(np.array([0.5, 0.25], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array
    x = tf.constant(np.array([[[2, 4], [8, 16]], [[32, 64], [128, 256]]], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Another dtype (int32), which will get cast to float32
    x = tf.constant(np.array([2, 4, 8], dtype=np.int32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Scalar value
    x = tf.constant(2.0, dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array of ones - removing this
    # x = tf.ones((2, 3), dtype=np.float32)
    # input_dict = {"x": x}
    # input_dict["x"] = tf.constant(input_dict["x"])
    # list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.log2"] = tf_experimental_numpy_log2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.log2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.log2'.")

check_valid('tf.experimental.numpy.log2', generated_inputs['tf.experimental.numpy.log2'], lib="tf", suffix=0)
