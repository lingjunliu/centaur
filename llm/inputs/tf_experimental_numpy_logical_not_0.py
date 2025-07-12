
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_logical_not_inputs():
    list_of_inputs = []

    # Input 1: Basic boolean array
    x = np.array([True, False, True, False])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer array (treated as boolean)
    x = np.array([1, 0, 2, -1])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float array (treated as boolean)
    x = np.array([1.0, 0.0, -1.0, 2.5])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D boolean array
    x = np.array([[True, False], [False, True]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D integer array
    x = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Array with mixed types (will be converted to boolean)
    x = np.array([True, 0, 1.0, False])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large array
    x = np.random.choice([True, False], size=(100,))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Array with negative values
    x = np.array([-1, -2, -3, 0, 1, 2])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty array
    x = np.array([])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multi-dimensional array with a mix of positive and negative numbers
    x = np.array([[[1, -1, 0], [0, 1, -1]], [[-1, 0, 1], [1, -1, 0]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.logical_not"] = tf_experimental_numpy_logical_not_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.logical_not' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.logical_not'.")

check_valid('tf.experimental.numpy.logical_not', generated_inputs['tf.experimental.numpy.logical_not'], lib="tf", suffix=0)
