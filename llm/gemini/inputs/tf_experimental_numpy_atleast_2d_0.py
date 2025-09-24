
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_experimental_numpy_atleast_2d_inputs():
    """
    Generates a list of valid inputs for the tf.experimental.numpy.atleast_2d function.
    """
    list_of_inputs = []

    # The error indicates that the testing framework cannot convert a NumPy array
    # of dtype=object into a TensorFlow tensor. This happens when trying to
    # represent a list of arrays with different shapes (e.g., [np.array(1), np.array([2,3])])
    # as a single NumPy array.
    # To resolve this, each input will consist of a single, regular NumPy array.
    # This effectively tests the single-argument version of atleast_2d, i.e.,
    # atleast_2d(ary), as it's the only way to satisfy the testing framework's
    # constraints of requiring an object with a .shape attribute that is also
    # convertible to a single tf.Tensor.

    # Input 1: Single scalar (0-D array)
    input_dict_1 = {'arys': np.array(5)}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Single 1-D array
    input_dict_2 = {'arys': np.array([1, 2, 3])}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Single 2-D array (should remain unchanged)
    input_dict_3 = {'arys': np.array([[1, 2], [3, 4]])}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Single 3-D array (should remain unchanged)
    input_dict_4 = {'arys': np.array([[[1], [2]], [[3], [4]]])}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Single negative float scalar
    input_dict_5 = {'arys': np.array(-100.5)}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 1-D array of floating-point numbers
    input_dict_6 = {'arys': np.array([-0.5, 0.0, 1.5e-5], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Empty 1-D array
    input_dict_7 = {'arys': np.array([])}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: A single 1-D array with one element
    input_dict_8 = {'arys': np.array([42])}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Boolean array
    input_dict_9 = {'arys': np.array([True, False, True])}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Higher-dimensional array
    input_dict_10 = {'arys': np.arange(24).reshape(2, 3, 4).astype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Empty 2D array
    input_dict_11 = {'arys': np.empty((2, 0), dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: Complex numbers
    input_dict_12 = {'arys': np.array([1+2j, 3+4j, 5+6j])}
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.atleast_2d"] = get_tf_experimental_numpy_atleast_2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.atleast_2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.atleast_2d'.")

check_valid('tf.experimental.numpy.atleast_2d', generated_inputs['tf.experimental.numpy.atleast_2d'], lib="tf", suffix=0)
