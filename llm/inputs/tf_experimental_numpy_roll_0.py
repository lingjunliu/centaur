
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_roll_inputs():
    list_of_inputs = []

    # Input 1
    a = tf.constant(np.array([1, 2, 3, 4, 5]))
    shift = 2
    axis = 0
    input_dict = {"a": a, "shift": shift, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = tf.constant(np.array([[1, 2], [3, 4]]))
    shift = 1
    axis = 0
    input_dict = {"a": a, "shift": shift, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = tf.constant(np.array([[1, 2], [3, 4]]))
    shift = -1
    axis = 1
    input_dict = {"a": a, "shift": shift, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    shift = 1
    axis = 1
    input_dict = {"a": a, "shift": shift, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    shift = -1
    axis = 2
    input_dict = {"a": a, "shift": shift, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = tf.constant(np.array([[1, 2, 3], [4, 5, 6]]))
    shift = 2
    axis = 1
    input_dict = {"a": a, "shift": shift, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    a = tf.constant(np.array([1, 2, 3, 4, 5]))
    shift = -2
    axis = 0
    input_dict = {"a": a, "shift": shift, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = tf.constant(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
    shift = 1
    axis = 0
    input_dict = {"a": a, "shift": shift, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = tf.constant(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
    shift = -1
    axis = 1
    input_dict = {"a": a, "shift": shift, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = tf.constant(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
    shift = 2
    axis = 2
    input_dict = {"a": a, "shift": shift, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    for input_dict in list_of_inputs:
        input_dict['a'] = input_dict['a'].numpy()

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.roll"] = tf_experimental_numpy_roll_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.roll' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.roll'.")

check_valid('tf.experimental.numpy.roll', generated_inputs['tf.experimental.numpy.roll'], lib="tf", suffix=0)
