
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_vdot_inputs():
    list_of_inputs = []

    # Input 1: Basic test with integer arrays
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([2, 3, 4, 5, 6])
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic test with float arrays
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 3.0, 4.0])
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Test with different shapes (must be flattened)
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Test with negative values
    a = np.array([-1, -2, -3])
    b = np.array([1, 2, 3])
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Test with a single element array
    a = np.array([5])
    b = np.array([10])
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Test with complex numbers
    a = np.array([1 + 1j, 2 + 2j])
    b = np.array([3 + 3j, 4 + 4j])
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Test with zero values
    a = np.array([0, 0, 0])
    b = np.array([1, 2, 3])
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Test with large numbers
    a = np.array([1e9, 2e9])
    b = np.array([3e9, 4e9])
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Test with multi-dimensional arrays of same shapes
    a = np.array([[1, 2, 3]])
    b = np.array([[4, 5, 6]])

    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Test with multi-dimensional arrays of same shapes
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.vdot"] = tf_experimental_numpy_vdot_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.vdot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.vdot'.")

check_valid('tf.experimental.numpy.vdot', generated_inputs['tf.experimental.numpy.vdot'], lib="tf", suffix=0)
