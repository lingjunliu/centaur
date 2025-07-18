
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_argsort_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array
    input_dict = {
        'a': np.array([3, 1, 4, 1, 5, 9, 2, 6]),
        'axis': -1,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, sort along axis 0
    input_dict = {
        'a': np.array([[10, 20, 5], [15, 0, 25]]),
        'axis': 0,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array with negative float values, sort along axis 1
    input_dict = {
        'a': np.array([[0., 3., -1.], [-2., 1., 3.]]),
        'axis': 1,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array with a negative axis
    a_3d = np.array([[[10, 11, 8, 9], [6, 7, 4, 5], [2, 3, 0, 1]],
                     [[22, 23, 20, 21], [18, 19, 16, 17], [14, 15, 12, 13]]])
    input_dict = {
        'a': a_3d,
        'axis': -2,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Using 'stable' kind for sorting with duplicates
    input_dict = {
        'a': np.array([5, 2, 6, 2, 7, 2, 8, 5]),
        'axis': -1,
        'kind': 'stable',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty array (edge case)
    input_dict = {
        'a': np.array([]).astype(np.int32),
        'axis': 0,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Array with all identical elements
    input_dict = {
        'a': np.full((3, 4), 5.0),
        'axis': 1,
        'kind': 'stable',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Array containing NaN and Inf values
    input_dict = {
        'a': np.array([1., np.nan, -np.inf, 3., np.inf, -1.]),
        'axis': -1,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D float array with duplicates
    input_dict = {
        'a': np.array([0.5, 0.2, 0.8, -0.1, 0.9, 0.2]),
        'axis': 0,
        'kind': 'quicksort',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single-element array
    input_dict = {
        'a': np.array([42]),
        'axis': -1,
        'kind': 'stable',
        'order': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.argsort"] = tf_experimental_numpy_argsort_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.argsort' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.argsort'.")

check_valid('tf.experimental.numpy.argsort', generated_inputs['tf.experimental.numpy.argsort'], lib="tf", suffix=0)
