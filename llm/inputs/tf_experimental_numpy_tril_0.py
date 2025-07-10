
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_tril_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D array with default k
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = 0
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array with k=1
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = 1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array with k=-1
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = -1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array with k greater than dimension
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = 5
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array with k less than negative dimension
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = -5
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, different k
    m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    k = 1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, different k
    m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    k = -1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 2D array with negative values
    m = np.array([[-1, 2, -3], [4, -5, 6], [-7, 8, -9]])
    k = 0
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 2D array of floats
    m = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]])
    k = 0
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Example from NumPy documentation
    m = np.arange(24).reshape(4,6)
    k = -1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.tril"] = tf_experimental_numpy_tril_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.tril' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.tril'.")

check_valid('tf.experimental.numpy.tril', generated_inputs['tf.experimental.numpy.tril'], lib="tf", suffix=0)
