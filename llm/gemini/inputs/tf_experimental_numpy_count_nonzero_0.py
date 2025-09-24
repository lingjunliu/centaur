
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_count_nonzero_inputs():
    list_of_inputs = []

    # Input 1: 1D array, axis=None
    a = np.array([0, 1, 2, 3, 4])
    axis = None
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, axis=0
    a = np.array([[0, 1, 2], [3, 0, 4]])
    axis = 0
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, axis=1
    a = np.array([[0, 1, 2], [3, 0, 4]])
    axis = 1
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, axis=0
    a = np.array([[[0, 1], [2, 3]], [[4, 0], [6, 7]]])
    axis = 0
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, axis=1
    a = np.array([[[0, 1], [2, 3]], [[4, 0], [6, 7]]])
    axis = 1
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, axis=2
    a = np.array([[[0, 1], [2, 3]], [[4, 0], [6, 7]]])
    axis = 2
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: array with negative values, axis=None
    a = np.array([-1, 0, 1, -2, 2])
    axis = None
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: array with all zeros, axis=None
    a = np.array([0, 0, 0, 0])
    axis = None
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: array with all non-zeros, axis=None
    a = np.array([1, 2, 3, 4])
    axis = None
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multi-dimensional array, axis = -1
    a = np.array([[1, 0, 2], [0, 3, 4]])
    axis = -1
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.count_nonzero"] = tf_experimental_numpy_count_nonzero_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.count_nonzero' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.count_nonzero'.")

check_valid('tf.experimental.numpy.count_nonzero', generated_inputs['tf.experimental.numpy.count_nonzero'], lib="tf", suffix=0)
