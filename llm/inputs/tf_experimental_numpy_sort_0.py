
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_sort_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D integer array
    input_dict_1 = {
        'a': np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int32),
        'axis': -1,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D float array, sort along columns (axis=0)
    input_dict_2 = {
        'a': np.array([[3.3, 1.1], [2.2, 4.4]], dtype=np.float32),
        'axis': 0,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D integer array, sort along rows (axis=1)
    input_dict_3 = {
        'a': np.array([[5, 2, 8], [1, 9, 4]], dtype=np.int32),
        'axis': 1,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 1D array with negative float values, use mergesort
    input_dict_4 = {
        'a': np.array([-1.5, 0.0, -5.5, 2.0, -2.5], dtype=np.float64),
        'axis': -1,
        'kind': 'mergesort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 3D array, sort along the last axis
    input_dict_5 = {
        'a': np.arange(24, dtype=np.float32).reshape((2, 3, 4)),
        'axis': -1,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 3D array, sort along a middle axis
    input_dict_6 = {
        'a': np.random.randint(-100, 100, size=(2, 4, 3), dtype=np.int32),
        'axis': 1,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Array with complex numbers
    input_dict_7 = {
        'a': np.array([1+2j, 3-1j, 1-2j, 0+0j], dtype=np.complex64),
        'axis': 0,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Empty array
    input_dict_8 = {
        'a': np.array([], dtype=np.float32),
        'axis': -1,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Single-element array
    input_dict_9 = {
        'a': np.array([42], dtype=np.int32),
        'axis': 0,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Array with duplicate values
    input_dict_10 = {
        'a': np.array([5, 2, 5, 1, 2, 5, 1], dtype=np.int32),
        'axis': -1,
        'kind': 'mergesort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: 2D array, already sorted along the given axis
    input_dict_11 = {
        'a': np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        'axis': 1,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: 2D array, reverse sorted along the given axis
    input_dict_12 = {
        'a': np.array([[8, 5, 1], [9, 6, 2]], dtype=np.int32),
        'axis': 1,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.sort"] = tf_experimental_numpy_sort_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.sort' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.sort'.")

check_valid('tf.experimental.numpy.sort', generated_inputs['tf.experimental.numpy.sort'], lib="tf", suffix=0)
