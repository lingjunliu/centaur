
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_atleast_1d_inputs():
    list_of_inputs = []

    # Input 1: Scalar
    input_dict = {"arys": [np.array(1)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array
    input_dict = {"arys": [np.array([1, 2, 3])]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array
    input_dict = {"arys": [np.array([[1, 2], [3, 4]])]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multiple arrays (scalar, 1D, 2D)
    input_dict = {"arys": [np.array(5), np.array([6, 7]), np.array([[8, 9], [10, 11]])]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty array
    input_dict = {"arys": [np.array([])]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Array with different data type (float)
    input_dict = {"arys": [np.array([1.1, 2.2, 3.3])]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Array with different data type (bool)
    input_dict = {"arys": [np.array([True, False, True])]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Array with negative values
    input_dict = {"arys": [np.array([-1, -2, -3])]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Array with zeros
    input_dict = {"arys": [np.array([0, 0, 0])]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array
    input_dict = {"arys": [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: empty list
    input_dict = {"arys": [np.array([]).reshape(0)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.atleast_1d"] = tf_experimental_numpy_atleast_1d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.atleast_1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.atleast_1d'.")

check_valid('tf.experimental.numpy.atleast_1d', generated_inputs['tf.experimental.numpy.atleast_1d'], lib="tf", suffix=0)
