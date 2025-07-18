
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_broadcast_arrays_inputs():
    list_of_inputs = []

    # Case 1: 1D array, unpacked into scalar arguments
    input_dict = {
        'args': np.array([1, 2, 3])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D array, unpacked into 1D array arguments
    input_dict = {
        'args': np.array([[10, 20], [30, 40]])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 3D array, unpacked into 2D array arguments
    input_dict = {
        'args': np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: A single array argument
    input_dict = {
        'args': np.array([[1, 2, 3, 4]])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Multiple scalar arguments with negative values
    input_dict = {
        'args': np.array([-1, -5, -10, -20])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Floating point values
    input_dict = {
        'args': np.array([[1.1, 2.2], [-3.3, -4.4]])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Unpacking into arrays of shape (1,)
    input_dict = {
        'args': np.array([[10], [20], [30]])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: A single scalar argument
    input_dict = {
        'args': np.array([100])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 9: Different integer dtype
    input_dict = {
        'args': np.array([[1, 2], [3, 4]], dtype=np.int16)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: Unpacking into empty arrays
    input_dict = {
        'args': np.empty((3, 0))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 11: Zeros
    input_dict = {
        'args': np.zeros((5, 2))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 12: Larger number of arrays to unpack
    input_dict = {
        'args': np.ones((10, 3))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.broadcast_arrays"] = tf_experimental_numpy_broadcast_arrays_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.broadcast_arrays' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.broadcast_arrays'.")

check_valid('tf.experimental.numpy.broadcast_arrays', generated_inputs['tf.experimental.numpy.broadcast_arrays'], lib="tf", suffix=0)
