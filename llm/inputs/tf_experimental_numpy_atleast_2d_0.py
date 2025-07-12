
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_atleast_2d_inputs():
    list_of_inputs = []

    # Input 1: Single 0-D array
    arys = [np.array(1)]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Single 1-D array
    arys = [np.array([1, 2, 3])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single 2-D array
    arys = [np.array([[1, 2], [3, 4]])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multiple 0-D arrays
    arys = [np.array(1), np.array(2)]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multiple 1-D arrays
    arys = [np.array([1, 2]), np.array([3, 4])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Mixed 0-D and 1-D arrays
    arys = [np.array(1), np.array([2, 3])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Mixed 1-D and 2-D arrays
    arys = [np.array([1, 2]), np.array([[3, 4], [5, 6]])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative values
    arys = [np.array([-1, -2])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty array with defined dtype
    arys = [np.array([], dtype=np.int32)]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multiple arrays with different dtypes
    arys = [np.array([1, 2], dtype=np.int32), np.array([3.0, 4.0], dtype=np.float32)]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.atleast_2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.atleast_2d'.")

check_valid('tf.experimental.numpy.atleast_2d', generated_inputs['tf.experimental.numpy.atleast_2d'], lib="tf", suffix=0)
