
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_diagonal_inputs():
    list_of_inputs = []

    # Input 1
    a = np.array([[1, 2], [3, 4]])
    offset = 0
    axis1 = 0
    axis2 = 1
    input_dict = {"a": a, "offset": offset, "axis1": axis1, "axis2": axis2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    offset = 1
    axis1 = 0
    axis2 = 1
    input_dict = {"a": a, "offset": offset, "axis1": axis1, "axis2": axis2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    offset = -1
    axis1 = 0
    axis2 = 1
    input_dict = {"a": a, "offset": offset, "axis1": axis1, "axis2": axis2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    offset = 0
    axis1 = 0
    axis2 = 1
    input_dict = {"a": a, "offset": offset, "axis1": axis1, "axis2": axis2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    offset = 1
    axis1 = 0
    axis2 = 2
    input_dict = {"a": a, "offset": offset, "axis1": axis1, "axis2": axis2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    offset = -1
    axis1 = 0
    axis2 = 2
    input_dict = {"a": a, "offset": offset, "axis1": axis1, "axis2": axis2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.array([[1, 2], [3, 4]])
    offset = 0
    axis1 = 1
    axis2 = 0
    input_dict = {"a": a, "offset": offset, "axis1": axis1, "axis2": axis2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    offset = 0
    axis1 = 1
    axis2 = 2
    input_dict = {"a": a, "offset": offset, "axis1": axis1, "axis2": axis2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    offset = 1
    axis1 = 1
    axis2 = 2
    input_dict = {"a": a, "offset": offset, "axis1": axis1, "axis2": axis2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    offset = -1
    axis1 = 1
    axis2 = 2
    input_dict = {"a": a, "offset": offset, "axis1": axis1, "axis2": axis2}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.diagonal"] = tf_experimental_numpy_diagonal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.diagonal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.diagonal'.")

check_valid('tf.experimental.numpy.diagonal', generated_inputs['tf.experimental.numpy.diagonal'], lib="tf", suffix=0)
