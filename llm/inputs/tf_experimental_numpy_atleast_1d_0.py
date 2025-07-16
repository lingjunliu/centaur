
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_atleast_1d_inputs():
    list_of_inputs = []

    # Input 1: Scalar
    arys = [np.array(1)]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array
    arys = [np.array([1, 2, 3])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array
    arys = [np.array([[1, 2], [3, 4]])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multiple scalars
    arys = [np.array(1), np.array(2)]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Mixed scalars and arrays
    arys = [np.array(1), np.array([2, 3])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty array (reshaped to 1D)
    arys = [np.array([]).reshape((0,))]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Array with negative values
    arys = [np.array([-1, -2, -3])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Array with zeros
    arys = [np.array([0, 0, 0])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Array with different data types (int and float)
    arys = [np.array([1, 2.5, 3])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multiple 2D arrays
    arys = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Single float
    arys = [np.array(3.14)]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.atleast_1d"] = tf_experimental_numpy_atleast_1d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.atleast_1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.atleast_1d'.")

check_valid('tf.experimental.numpy.atleast_1d', generated_inputs['tf.experimental.numpy.atleast_1d'], lib="tf", suffix=0)
