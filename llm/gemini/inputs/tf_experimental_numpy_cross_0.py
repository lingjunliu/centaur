
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_cross_inputs():
    list_of_inputs = []

    # Input 1
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    axisa = -1
    axisb = -1
    axisc = -1
    axis = None

    input_dict = {
        "a": a,
        "b": b,
        "axisa": axisa,
        "axisb": axisb,
        "axisc": axisc,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[4, 5, 6], [1, 2, 3]])
    axisa = -1
    axisb = -1
    axisc = -1
    axis = None
    input_dict = {
        "a": a,
        "b": b,
        "axisa": axisa,
        "axisb": axisb,
        "axisc": axisc,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[4, 5, 6], [1, 2, 3]])
    axisa = 0
    axisb = 0
    axisc = 0
    axis = None

    input_dict = {
        "a": a,
        "b": b,
        "axisa": axisa,
        "axisb": axisb,
        "axisc": axisc,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[4, 5, 6], [1, 2, 3]])
    axisa = 1
    axisb = 1
    axisc = 1
    axis = None

    input_dict = {
        "a": a,
        "b": b,
        "axisa": axisa,
        "axisb": axisb,
        "axisc": axisc,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    b = np.array([[[4, 5, 6], [1, 2, 3]], [[10, 11, 12], [7, 8, 9]]])
    axisa = -1
    axisb = -1
    axisc = -1
    axis = None
    input_dict = {
        "a": a,
        "b": b,
        "axisa": axisa,
        "axisb": axisb,
        "axisc": axisc,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    b = np.array([[[4, 5, 6], [1, 2, 3]], [[10, 11, 12], [7, 8, 9]]])
    axisa = 0
    axisb = 0
    axisc = 0
    axis = None
    input_dict = {
        "a": a,
        "b": b,
        "axisa": axisa,
        "axisb": axisb,
        "axisc": axisc,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    b = np.array([[[4, 5, 6], [1, 2, 3]], [[10, 11, 12], [7, 8, 9]]])
    axisa = 1
    axisb = 1
    axisc = 1
    axis = None
    input_dict = {
        "a": a,
        "b": b,
        "axisa": axisa,
        "axisb": axisb,
        "axisc": axisc,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    b = np.array([[[4, 5, 6], [1, 2, 3]], [[10, 11, 12], [7, 8, 9]]])
    axisa = 2
    axisb = 2
    axisc = 2
    axis = None
    input_dict = {
        "a": a,
        "b": b,
        "axisa": axisa,
        "axisb": axisb,
        "axisc": axisc,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    axisa = -1
    axisb = -1
    axisc = -1
    axis = None
    input_dict = {
        "a": a,
        "b": b,
        "axisa": axisa,
        "axisb": axisb,
        "axisc": axisc,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = np.array([0, 1, 0])
    b = np.array([0, 0, 1])
    axisa = -1
    axisb = -1
    axisc = -1
    axis = None
    input_dict = {
        "a": a,
        "b": b,
        "axisa": axisa,
        "axisb": axisb,
        "axisc": axisc,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[4, 5, 6], [1, 2, 3]])
    axisa = -1
    axisb = -1
    axisc = -1
    axis = None
    a[0, 1] = -2
    b[0, 1] = -5
    input_dict = {
        "a": a,
        "b": b,
        "axisa": axisa,
        "axisb": axisb,
        "axisc": axisc,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.cross"] = tf_experimental_numpy_cross_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.cross' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.cross'.")

check_valid('tf.experimental.numpy.cross', generated_inputs['tf.experimental.numpy.cross'], lib="tf", suffix=0)
