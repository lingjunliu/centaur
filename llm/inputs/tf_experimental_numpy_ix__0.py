
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_ix_inputs():
    list_of_inputs = []

    # Input 1: Basic case with 1D arrays
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    input_dict = {"args": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Two 1D arrays
    a = np.array([0, 1])
    b = np.array([0, 1, 2])
    input_dict = {"args": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Three 1D arrays
    a = np.array([0, 1])
    b = np.array([0, 1, 2])
    c = np.array([0, 1])
    input_dict = {"args": [a, b, c]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: One array
    a = np.array([0, 1, 2])
    input_dict = {"args": [a]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty arrays
    a = np.array([])
    input_dict = {"args": [a]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty and non empty array.
    a = np.array([])
    b = np.array([1,2,3])
    input_dict = {"args": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Two empty arrays
    a = np.array([])
    b = np.array([])
    input_dict = {"args": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Three empty arrays
    a = np.array([])
    b = np.array([])
    c = np.array([])
    input_dict = {"args": [a, b, c]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Non-empty arrays with different datatypes.
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([4, 5, 6], dtype=np.float64)
    input_dict = {"args": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Arrays with negative numbers
    a = np.array([-1, 0, 1])
    b = np.array([-2, -1, 0])
    input_dict = {"args": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.ix_"] = tf_experimental_numpy_ix_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.ix_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.ix_'.")

check_valid('tf.experimental.numpy.ix_', generated_inputs['tf.experimental.numpy.ix_'], lib="tf", suffix=0)
