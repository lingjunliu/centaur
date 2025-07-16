
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_ix_inputs():
    list_of_inputs = []

    # Input 1: Basic case with two 1D arrays
    a = np.array([1, 2, 3])
    b = np.array([4, 5])
    input_dict = {"args": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Three 1D arrays
    a = np.array([1, 2])
    b = np.array([3, 4, 5])
    c = np.array([6])
    input_dict = {"args": [a, b, c]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: One empty array
    a = np.array([])
    b = np.array([1, 2])
    input_dict = {"args": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    # Input 5: One array
    a = np.array([1, 2, 3])
    input_dict = {"args": [a]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Arrays with different dtypes (int, float) - Changed to same dtype
    a = np.array([1, 2, 3])
    b = np.array([4, 5])
    input_dict = {"args": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Arrays with larger dimensions
    a = np.array([[1, 2], [3, 4]])
    b = np.array([0, 1])
    input_dict = {"args": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Three arrays, different sizes and shapes
    a = np.array([1, 2])
    b = np.array([[3, 4], [5, 6]])
    c = np.array([7])
    input_dict = {"args": [a, b, c]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9:  Negative values
    a = np.array([-1, 2])
    b = np.array([3, -4, 5])
    input_dict = {"args": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: One array with a single element
    a = np.array([5])
    b = np.array([1, 2, 3])
    input_dict = {"args": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: all empty arrays. Removed because this makes ix_ return an error.
    # a = np.array([])
    # b = np.array([])
    # input_dict = {"args": [a,b]}
    # list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: 3D array
    a = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
    b = np.array([0,1])
    input_dict = {"args": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13: boolean array
    a = np.array([True, False, True])
    b = np.array([False, True])
    input_dict = {"args": [a,b]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.ix_"] = tf_experimental_numpy_ix_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.ix_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.ix_'.")

check_valid('tf.experimental.numpy.ix_', generated_inputs['tf.experimental.numpy.ix_'], lib="tf", suffix=0)
