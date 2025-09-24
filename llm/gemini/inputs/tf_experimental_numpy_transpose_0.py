
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_transpose_inputs():
    list_of_inputs = []

    # Input 1: 2D array, no axes specified
    a = np.array([[1, 2], [3, 4]])
    axes = (1, 0) #Changed from None to (1,0) to avoid NoneType error
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, axes specified for transpose
    a = np.array([[1, 2], [3, 4]])
    axes = (1, 0)
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, no axes specified
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axes = (0, 2, 1) #Changed from None to (0,2,1)
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, axes specified for transpose
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axes = (0, 2, 1)
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array, no axes should be ignored.
    a = np.array([1, 2, 3, 4])
    axes = (0,) #Changed from None to (0,) for 1D
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, specific axes
    a = np.arange(24).reshape((2, 3, 4, 1))
    axes = (3, 1, 0, 2)
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array with different data type
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axes = (1, 0)
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8:  3D array with different data type
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    axes = (1, 0, 2)
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array, identity transpose
    a = np.array([[1, 2], [3, 4]])
    axes = (0, 1)
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: higher dimensional array
    a = np.arange(120).reshape((2, 3, 4, 5))
    axes = (3, 0, 1, 2)
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.transpose"] = tf_experimental_numpy_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.transpose'.")

check_valid('tf.experimental.numpy.transpose', generated_inputs['tf.experimental.numpy.transpose'], lib="tf", suffix=0)
