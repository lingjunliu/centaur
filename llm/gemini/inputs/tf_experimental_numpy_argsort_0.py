
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_argsort_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array
    a1 = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    input_dict1 = {
        'a': a1.astype(np.int32),
        'axis': -1,
        'kind': 'quicksort',
        'order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D array, sort along axis 0
    a2 = np.array([[3, 2, 1], [6, 5, 4]])
    input_dict2 = {
        'a': a2.astype(np.int32),
        'axis': 0,
        'kind': 'mergesort',
        'order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D array, sort along axis 1
    a3 = np.array([[9, 1, 7], [3, 8, 2]])
    input_dict3 = {
        'a': a3.astype(np.int32),
        'axis': 1,
        'kind': 'heapsort',
        'order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D array with negative values
    a4 = np.array([-5, 0, -10, 2, -1])
    input_dict4 = {
        'a': a4.astype(np.int32),
        'axis': 0,
        'kind': 'stable',
        'order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D array with duplicate values and stable sort
    a5 = np.array([[5, 2, 5], [3, 3, 1]])
    input_dict5 = {
        'a': a5.astype(np.int32),
        'axis': -1,
        'kind': 'stable',
        'order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Floating point array
    a6 = np.array([1.1, 9.9, 2.2, 5.5, 1.1, 0.5])
    input_dict6 = {
        'a': a6.astype(np.float32),
        'axis': -1,
        'kind': 'quicksort',
        'order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 3D array, sort along axis 1
    a7 = np.arange(24).reshape(2, 4, 3)
    np.random.shuffle(a7.ravel())
    a7 = a7.reshape(2, 4, 3)
    input_dict7 = {
        'a': a7.astype(np.int64),
        'axis': 1,
        'kind': 'mergesort',
        'order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: 3D array, sort along a negative axis
    a8 = np.random.rand(3, 2, 4)
    input_dict8 = {
        'a': a8.astype(np.float64),
        'axis': -2,
        'kind': 'heapsort',
        'order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Array with all elements being the same
    a9 = np.ones((4, 5))
    input_dict9 = {
        'a': a9.astype(np.float32),
        'axis': 0,
        'kind': 'stable',
        'order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Empty array with valid rank for axis
    a10 = np.array([], dtype=np.int32).reshape(1,0)
    input_dict10 = {
        'a': a10,
        'axis': 0,
        'kind': 'quicksort',
        'order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    # Input 11: Single element array
    a11 = np.array([42])
    input_dict11 = {
        'a': a11.astype(np.int32),
        'axis': -1,
        'kind': 'quicksort',
        'order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict11))

    # Input 12: A 4D array
    a12 = np.random.uniform(-100, 100, size=(2,2,2,2))
    input_dict12 = {
        'a': a12.astype(np.float32),
        'axis': 3,
        'kind': 'stable',
        'order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict12))

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
