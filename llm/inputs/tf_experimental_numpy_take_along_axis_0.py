
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.experimental.numpy.experimental_enable_numpy_behavior()

def tf_experimental_numpy_take_along_axis_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array
    arr = tf.constant(np.array([1, 2, 3, 4, 5]))
    indices = tf.constant(np.array([0, 2, 4]))
    axis = 0
    input_dict = {"arr": arr, "indices": indices, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, axis 0
    arr = tf.constant(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    indices = tf.constant(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]))
    axis = 0
    input_dict = {"arr": arr, "indices": indices, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, axis 1
    arr = tf.constant(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    indices = tf.constant(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]))
    axis = 1
    input_dict = {"arr": arr, "indices": indices, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, negative indices, axis 0
    arr = tf.constant(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    indices = tf.constant(np.array([[0, 0, 0], [-1, -1, -1], [-2, -2, -2]]))
    axis = 0
    input_dict = {"arr": arr, "indices": indices, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, negative indices, axis 1
    arr = tf.constant(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    indices = tf.constant(np.array([[0, -1, -2], [0, -1, -2], [0, -1, -2]]))
    axis = 1
    input_dict = {"arr": arr, "indices": indices, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, axis 0
    arr = tf.constant(np.arange(27).reshape((3, 3, 3)))
    indices = tf.constant(np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[2, 2, 2], [2, 2, 2], [2, 2, 2]]]))
    axis = 0
    input_dict = {"arr": arr, "indices": indices, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, axis 1
    arr = tf.constant(np.arange(27).reshape((3, 3, 3)))
    indices = tf.constant(np.array([[[0, 0, 0], [1, 1, 1], [2, 2, 2]], [[0, 0, 0], [1, 1, 1], [2, 2, 2]], [[0, 0, 0], [1, 1, 1], [2, 2, 2]]]))
    axis = 1
    input_dict = {"arr": arr, "indices": indices, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array, axis 2
    arr = tf.constant(np.arange(27).reshape((3, 3, 3)))
    indices = tf.constant(np.array([[[0, 1, 2], [0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]]]))
    axis = 2
    input_dict = {"arr": arr, "indices": indices, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array with negative axis
    arr = tf.constant(np.array([1, 2, 3, 4, 5]))
    indices = tf.constant(np.array([0, 2, 4]))
    axis = -1
    input_dict = {"arr": arr, "indices": indices, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, axis 0, different indices
    arr = tf.constant(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    indices = tf.constant(np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]]))
    axis = 0
    input_dict = {"arr": arr, "indices": indices, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.take_along_axis"] = tf_experimental_numpy_take_along_axis_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.take_along_axis' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.take_along_axis'.")

check_valid('tf.experimental.numpy.take_along_axis', generated_inputs['tf.experimental.numpy.take_along_axis'], lib="tf", suffix=0)
