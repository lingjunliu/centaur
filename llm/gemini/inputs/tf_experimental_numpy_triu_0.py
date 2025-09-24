
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_triu_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D array, k=0
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = 0
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(input_dict)

    # Input 2: 2D array, k=1
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = 1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(input_dict)

    # Input 3: 2D array, k=-1
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = -1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(input_dict)

    # Input 4: 3D array, k=0
    m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    k = 0
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(input_dict)

    # Input 5: 3D array, k=1
    m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    k = 1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(input_dict)

    # Input 6: 3D array, k=-1
    m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    k = -1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(input_dict)

    # Input 7: Larger 2D array, k=2
    m = np.random.rand(5, 5)
    k = 2
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(input_dict)

    # Input 8: Larger 2D array, k=-2
    m = np.random.rand(5, 5)
    k = -2
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(input_dict)

    # Input 9: Array with negative values, k=0
    m = np.array([[-1, 2, -3], [4, -5, 6], [-7, 8, -9]])
    k = 0
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(input_dict)

    # Input 10: Array with zeros, k=0
    m = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    k = 0
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.triu"] = tf_experimental_numpy_triu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.triu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.triu'.")

check_valid('tf.experimental.numpy.triu', generated_inputs['tf.experimental.numpy.triu'], lib="tf", suffix=0)
