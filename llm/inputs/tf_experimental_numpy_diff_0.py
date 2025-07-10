
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_diff_inputs():
    list_of_inputs = []

    # Input 1
    a = tf.constant(np.array([1, 2, 4, 7, 0]))
    n = 1
    axis = 0
    input_dict = {"a": a, "n": n, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = tf.constant(np.array([[1, 3, 6, 10, 15], [0, 2, 5, 9, 14]]))
    n = 2
    axis = 1
    input_dict = {"a": a, "n": n, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = tf.constant(np.array([[1, 3, 6, 10, 15], [0, 2, 5, 9, 14]]))
    n = 1
    axis = 0
    input_dict = {"a": a, "n": n, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = tf.constant(np.array([1, 2, 4, 7, 0]))
    n = 0
    axis = 0
    input_dict = {"a": a, "n": n, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = tf.constant(np.array([1, 2, 4, 7, 0]))
    n = 3
    axis = 0
    input_dict = {"a": a, "n": n, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    n = 1
    axis = 0
    input_dict = {"a": a, "n": n, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    n = 1
    axis = 1
    input_dict = {"a": a, "n": n, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    n = 1
    axis = 2
    input_dict = {"a": a, "n": n, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = tf.constant(np.array([1, 2, 4, 7, 0]), dtype=tf.float32)
    n = 1
    axis = 0
    input_dict = {"a": a, "n": n, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = tf.constant(np.array([[1, 3, 6, 10, 15], [0, 2, 5, 9, 14]]), dtype=tf.int64)
    n = 2
    axis = -1
    input_dict = {"a": a, "n": n, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.diff"] = tf_experimental_numpy_diff_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.diff' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.diff'.")

check_valid('tf.experimental.numpy.diff', generated_inputs['tf.experimental.numpy.diff'], lib="tf", suffix=0)
