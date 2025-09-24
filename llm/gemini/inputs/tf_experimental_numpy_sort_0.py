
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_sort_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array of integers
    input_dict1 = {
        'a': np.array([3, 1, 4, 1, 5, 9, 2, 6]),
        'axis': -1,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D array, sort along rows (axis=1)
    input_dict2 = {
        'a': np.array([[3, 1, 4], [1, 5, 9]]),
        'axis': 1,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D array, sort along columns (axis=0)
    input_dict3 = {
        'a': np.array([[3, 1, 4], [1, 5, 9]]),
        'axis': 0,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 2D array with negative numbers and floats, using negative axis
    input_dict4 = {
        'a': np.array([[-3.0, 1.5, -4.2], [1.1, -5.9, 9.0]]),
        'axis': -1,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D array, sort along a middle axis
    input_dict5 = {
        'a': np.random.randint(-10, 10, size=(2, 4, 3)).astype(np.float32),
        'axis': 1,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Stable sort on an array with repeated elements
    input_dict6 = {
        'a': np.array([9, 4, 9, 4, 9, 1, 4]),
        'axis': -1,
        'kind': 'stable',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Empty array (2D)
    input_dict7 = {
        'a': np.array([[]], dtype=np.float32),
        'axis': -1,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Array with a single element
    input_dict8 = {
        'a': np.array([[[-42]]]),
        'axis': 0,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Boolean array
    input_dict9 = {
        'a': np.array([[True, False], [False, True]]),
        'axis': 1,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Already sorted array
    input_dict10 = {
        'a': np.array([1, 2, 3, 4, 5, 6]),
        'axis': -1,
        'kind': 'stable',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    # Input 11: 1D array with axis=0
    input_dict11 = {
        'a': np.array([10, -1, 5, 0]),
        'axis': 0,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict11))

    # Input 12: Array with inf and nan
    input_dict12 = {
        'a': np.array([np.inf, 1.0, np.nan, -np.inf]),
        'axis': -1,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict12))

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
