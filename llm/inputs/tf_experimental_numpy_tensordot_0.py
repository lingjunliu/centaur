
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_tensordot_inputs():
    list_of_inputs = []

    # Input 1
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    axes = 1
    input_dict = {"a": tf.constant(a), "b": tf.constant(b), "axes": int(axes)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = np.array([[9, 10], [11, 12]])
    axes = 1
    input_dict = {"a": tf.constant(a), "b": tf.constant(b), "axes": int(axes)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    axes = 0
    input_dict = {"a": tf.constant(a), "b": tf.constant(b), "axes": int(axes)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[7, 8], [9, 10], [11, 12]])
    axes = 1
    input_dict = {"a": tf.constant(a), "b": tf.constant(b), "axes": int(axes)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    axes = 2
    input_dict = {"a": tf.constant(a), "b": tf.constant(b), "axes": int(axes)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    axes = 0
    input_dict = {"a": tf.constant(a), "b": tf.constant(b), "axes": int(axes)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([6, 7, 8, 9, 10])
    axes = 1
    input_dict = {"a": tf.constant(a), "b": tf.constant(b), "axes": int(axes)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = np.array([9, 10, 11, 12])
    axes = 0
    input_dict = {"a": tf.constant(a), "b": tf.constant(b), "axes": int(axes)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([7,8,9])
    axes = 0
    input_dict = {"a": tf.constant(a), "b": tf.constant(b), "axes": int(axes)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = np.array([1, 2])
    b = np.array([[3,4],[5,6]])
    axes = 0
    input_dict = {"a": tf.constant(a), "b": tf.constant(b), "axes": int(axes)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.tensordot"] = tf_experimental_numpy_tensordot_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.tensordot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.tensordot'.")

check_valid('tf.experimental.numpy.tensordot', generated_inputs['tf.experimental.numpy.tensordot'], lib="tf", suffix=0)
