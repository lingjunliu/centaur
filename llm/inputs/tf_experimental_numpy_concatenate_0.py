
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_concatenate_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D concatenation
    input_dict = {
        'arys': np.array([np.array([1, 2, 3]), np.array([4, 5, 6])]),
        'axis': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Concatenate three 2D arrays along axis 0
    input_dict = {
        'arys': np.array([np.ones((2, 3)), np.zeros((2, 3)), np.ones((2, 3)) * 5]),
        'axis': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Concatenate 2D arrays along axis 1
    input_dict = {
        'arys': np.array([np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]),
        'axis': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Concatenate 2D arrays with a negative axis
    input_dict = {
        'arys': np.array([np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]])]),
        'axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Concatenate arrays of different compatible dtypes
    input_dict = {
        'arys': np.array([np.array([1, 2], dtype=np.int32), np.array([3, 4], dtype=np.float32)]),
        'axis': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Concatenate 3D arrays along axis 0
    input_dict = {
        'arys': np.array([np.arange(8, dtype=np.float32).reshape(2, 2, 2), np.arange(8, 16, dtype=np.float32).reshape(2, 2, 2)]),
        'axis': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Concatenate 3D arrays along axis 1
    input_dict = {
        'arys': np.array([np.ones((2, 2, 2)), np.zeros((2, 2, 2))]),
        'axis': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Concatenate 3D arrays along axis 2
    input_dict = {
        'arys': np.array([np.ones((2, 2, 2)), np.zeros((2, 2, 2))]),
        'axis': 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Concatenate with empty arrays (of same shape)
    input_dict = {
        'arys': np.array([np.empty(shape=(2, 3), dtype=np.int64), np.empty(shape=(2, 3), dtype=np.int64)]),
        'axis': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Concatenate 4D arrays
    input_dict = {
        'arys': np.array([np.zeros((1, 2, 3, 4)), np.ones((1, 2, 3, 4))]),
        'axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Concatenating a single array in the sequence
    input_dict = {
        'arys': np.array([np.ones((2, 3))]),
        'axis': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Concatenate along an axis of size 1
    input_dict = {
        'arys': np.array([np.zeros((2, 1, 3)), np.ones((2, 1, 3))]),
        'axis': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.concatenate"] = tf_experimental_numpy_concatenate_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.concatenate' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.concatenate'.")

check_valid('tf.experimental.numpy.concatenate', generated_inputs['tf.experimental.numpy.concatenate'], lib="tf", suffix=0)
