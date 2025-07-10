
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_compress_inputs():
    list_of_inputs = []

    # Input 1: 1D array, basic test
    condition = tf.constant([True, False, True, False])
    a = tf.constant([1, 2, 3, 4])
    axis = None
    input_dict = {"condition": condition, "a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, axis=0
    condition = tf.constant([True, False, True])
    a = tf.constant([[1, 2], [3, 4], [5, 6]])
    axis = 0
    input_dict = {"condition": condition, "a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, axis=1
    condition = tf.constant([True, False])
    a = tf.constant([[1, 2], [3, 4]])
    axis = 1
    input_dict = {"condition": condition, "a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, axis=0
    condition = tf.constant([True, False])
    a = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = 0
    input_dict = {"condition": condition, "a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, axis=1
    condition = tf.constant([True, False])
    a = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = 1
    input_dict = {"condition": condition, "a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, axis=2
    condition = tf.constant([True, False])
    a = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = 2
    input_dict = {"condition": condition, "a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array, all False
    condition = tf.constant([False, False, False])
    a = tf.constant([1, 2, 3])
    axis = None
    input_dict = {"condition": condition, "a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array, all True
    condition = tf.constant([True, True, True])
    a = tf.constant([1, 2, 3])
    axis = None
    input_dict = {"condition": condition, "a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array, axis=0, empty condition
    condition = tf.constant([False, False])
    a = tf.constant([[1, 2], [3, 4]])
    axis = 0
    input_dict = {"condition": condition, "a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, axis=1, empty condition
    condition = tf.constant([False])
    a = tf.constant([[1, 2]])
    axis = 1
    input_dict = {"condition": condition, "a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.compress"] = tf_experimental_numpy_compress_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.compress' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.compress'.")

check_valid('tf.experimental.numpy.compress', generated_inputs['tf.experimental.numpy.compress'], lib="tf", suffix=0)
