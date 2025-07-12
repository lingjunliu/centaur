
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_flip_inputs():
    list_of_inputs = []

    # Input 1
    m = np.array([[1, 2], [3, 4]])
    axis = 0
    input_dict = {"m": m, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    m = np.array([[1, 2], [3, 4]])
    axis = 1
    input_dict = {"m": m, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    m = np.array([1, 2, 3, 4])
    axis = 0
    input_dict = {"m": m, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = 0
    input_dict = {"m": m, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = 1
    input_dict = {"m": m, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = 2
    input_dict = {"m": m, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = -1
    input_dict = {"m": m, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = -2
    input_dict = {"m": m, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = -3
    input_dict = {"m": m, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    m = np.array([1, 2, 3])
    axis = -1
    input_dict = {"m": m, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.flip"] = tf_experimental_numpy_flip_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.flip' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.flip'.")

check_valid('tf.experimental.numpy.flip', generated_inputs['tf.experimental.numpy.flip'], lib="tf", suffix=0)
