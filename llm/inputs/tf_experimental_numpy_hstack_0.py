
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_hstack_inputs():
    list_of_inputs = []

    # Input 1: Tuple of 1D arrays
    tup = (np.array([1, 2, 3]), np.array([4, 5, 6]))
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tuple of 2D arrays
    tup = (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tuple of single element arrays
    tup = (np.array([1]), np.array([2]))
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tuple containing only one array
    tup = (np.array([[1, 2], [3, 4]]),)
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tuple with a scalar converted to array
    tup = (np.array([5]), np.array([6]))
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tuple with negative values
    tup = (np.array([-1, -2, -3]), np.array([-4, -5, -6]))
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tuple of boolean arrays
    tup = (np.array([True, False, True]), np.array([False, True, False]))
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tuple of arrays with different data types, but same shape
    tup = (np.array([1, 2, 3], dtype=np.int32), np.array([4, 5, 6], dtype=np.int32))
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Tuple of arrays with different shapes
    tup = (np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8], [9, 10]]))
    input_dict = {"tup": tup}
    
    # Input 10: Tuple of arrays with one array is empty
    tup = (np.array([[1, 2], [3, 4]]), np.array([]))
    input_dict = {"tup": tup}
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.hstack"] = tf_experimental_numpy_hstack_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.hstack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.hstack'.")

check_valid('tf.experimental.numpy.hstack', generated_inputs['tf.experimental.numpy.hstack'], lib="tf", suffix=0)
