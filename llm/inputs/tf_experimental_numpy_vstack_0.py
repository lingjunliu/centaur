
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_vstack_inputs():
    list_of_inputs = []

    # Input 1: Tuple of 1D arrays
    tup = (np.array([1, 2, 3]), np.array([4, 5, 6]))
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tuple of 2D arrays
    tup = (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tuple of arrays with different shapes but compatible for vstack
    tup = (np.array([[1, 2, 3]]), np.array([[4, 5, 6]]))
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))


    # Input 5: Tuple of 3D arrays
    tup = (np.array([[[1, 2], [3, 4]]]), np.array([[[9, 10], [11, 12]]]))
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tuple of single element arrays
    tup = (np.array([1]), np.array([2]))
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tuple of arrays with negative values
    tup = (np.array([-1, -2, -3]), np.array([-4, -5, -6]))
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tuple of arrays with zeros
    tup = (np.array([0, 0, 0]), np.array([0, 0, 0]))
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tuple of arrays with a mix of positive and negative values
    tup = (np.array([1, -2, 3]), np.array([-4, 5, -6]))
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Tuple of same array multiple times
    arr = np.array([1,2,3])
    tup = (arr,arr,arr)
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.vstack"] = tf_experimental_numpy_vstack_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.vstack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.vstack'.")

check_valid('tf.experimental.numpy.vstack', generated_inputs['tf.experimental.numpy.vstack'], lib="tf", suffix=0)
