
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_argmax_inputs():
    list_of_inputs = []

    # Input 1: 1D array, no axis specified
    a = np.array([1, 5, 2, 8, 3])
    axis = None
    input_dict = {"a": tf.convert_to_tensor(a), "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, axis=0
    a = np.array([[1, 5, 2], [8, 3, 9]])
    axis = 0
    input_dict = {"a": tf.convert_to_tensor(a), "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, axis=1
    a = np.array([[1, 5, 2], [8, 3, 9]])
    axis = 1
    input_dict = {"a": tf.convert_to_tensor(a), "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, axis=0
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = 0
    input_dict = {"a": tf.convert_to_tensor(a), "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, axis=1
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = 1
    input_dict = {"a": tf.convert_to_tensor(a), "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, axis=2
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = 2
    input_dict = {"a": tf.convert_to_tensor(a), "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array with negative values, no axis specified
    a = np.array([-1, -5, -2, -8, -3])
    axis = None
    input_dict = {"a": tf.convert_to_tensor(a), "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array with negative and positive values, axis=0
    a = np.array([[-1, 5, -2], [8, -3, 9]])
    axis = 0
    input_dict = {"a": tf.convert_to_tensor(a), "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: array with only one element
    a = np.array([5])
    axis = None
    input_dict = {"a": tf.convert_to_tensor(a), "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array with same values in some rows, axis=1
    a = np.array([[1, 1, 1], [8, 3, 9]])
    axis = 1
    input_dict = {"a": tf.convert_to_tensor(a), "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.argmax"] = tf_experimental_numpy_argmax_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.argmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.argmax'.")

check_valid('tf.experimental.numpy.argmax', generated_inputs['tf.experimental.numpy.argmax'], lib="tf", suffix=0)
