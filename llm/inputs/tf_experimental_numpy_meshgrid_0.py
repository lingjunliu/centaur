
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_meshgrid_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    input_dict = {"xi": [tf.constant(x), tf.constant(y)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays with different lengths
    x = np.array([1, 2])
    y = np.array([3, 4, 5])
    input_dict = {"xi": [tf.constant(x), tf.constant(y)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D arrays with negative values
    x = np.array([-1, 0, 1])
    y = np.array([-2, 2])
    input_dict = {"xi": [tf.constant(x), tf.constant(y)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D arrays with float values
    x = np.array([1.0, 2.5, 3.0])
    y = np.array([4.2, 5.0])
    input_dict = {"xi": [tf.constant(x), tf.constant(y)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Three 1D arrays
    x = np.array([1, 2])
    y = np.array([3, 4])
    z = np.array([5, 6])
    input_dict = {"xi": [tf.constant(x), tf.constant(y), tf.constant(z)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Another simple valid case
    x = np.array([4, 5])
    y = np.array([6, 7])
    input_dict = {"xi": [tf.constant(x), tf.constant(y)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Only one element
    x = np.array([1])
    y = np.array([2])
    input_dict = {"xi": [tf.constant(x), tf.constant(y)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float and int arrays
    x = np.array([1.0, 2.0])
    y = np.array([3, 4])
    input_dict = {"xi": [tf.constant(x), tf.constant(y)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: More values
    x = np.array([1, 2, 3, 4])
    y = np.array([5, 6, 7, 8])
    input_dict = {"xi": [tf.constant(x), tf.constant(y)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.meshgrid"] = tf_experimental_numpy_meshgrid_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.meshgrid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.meshgrid'.")

check_valid('tf.experimental.numpy.meshgrid', generated_inputs['tf.experimental.numpy.meshgrid'], lib="tf", suffix=0)
