
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_vstack_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D arrays
    tup = (np.array([[1, 2, 3]]), np.array([[4, 5, 6]]))
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shaped 1D arrays
    tup = (np.array([1, 2]), np.array([3, 4]))
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Only 2D arrays, more than two
    tup = (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Arrays with negative values
    tup = (np.array([-1, -2, -3]), np.array([-4, -5, -6]))
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single array as a tuple
    tup = (np.array([[1, 2, 3]]),)
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Reshaping to 2D equivalent
    tup = (np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]]))
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Arrays with different data types (will be cast to a common type)
    tup = (np.array([1, 2, 3], dtype=np.int32), np.array([4, 5, 6], dtype=np.int32))
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different data types and shapes
    tup = (np.array([[1, 2]]), np.array([[3, 4], [5, 6]]))
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Ensuring that it can handle more dimensions.
    tup = (np.array([[[1, 2], [3, 4]]]), np.array([[[5, 6], [7, 8]]]))
    input_dict = {"tup": tup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single dimension with different size.
    tup = (np.array([1, 2, 3]), np.array([4, 5, 6]))
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
