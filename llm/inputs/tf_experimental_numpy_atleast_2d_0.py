
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_atleast_2d_inputs():
    list_of_inputs = []

    # Input 1: Single 0D array
    arys = [np.array(1)]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Single 1D array
    arys = [np.array([1, 2, 3])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single 2D array (already 2D)
    arys = [np.array([[1, 2], [3, 4]])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multiple 0D arrays
    arys = [np.array([1]), np.array([2])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multiple 1D arrays
    arys = [np.array([1, 2]), np.array([3, 4])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Mixed 0D and 1D arrays
    arys = [np.array([1]), np.array([2, 3])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Mixed 1D and 2D arrays
    arys = [np.array([1, 2]), np.array([[3, 4], [5, 6]])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single 3D array
    arys = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Multiple arrays with different shapes and dtypes (int and float)
    arys = [np.array([1.0]), np.array([2, 3])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty array
    arys = [np.array([])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.atleast_2d"] = tf_experimental_numpy_atleast_2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.atleast_2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.atleast_2d'.")

check_valid('tf.experimental.numpy.atleast_2d', generated_inputs['tf.experimental.numpy.atleast_2d'], lib="tf", suffix=0)
