
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_broadcast_arrays_inputs():
    list_of_inputs = []

    # Input 1: Two scalars
    input_dict = {"args": [np.array(1), np.array(2)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar and 1D array
    input_dict = {"args": [np.array(5), np.array([1, 2, 3])]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D and 2D array
    input_dict = {"args": [np.array([1, 2, 3]), np.array([[1], [2], [3]])]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Two 2D arrays with compatible shapes
    input_dict = {"args": [np.array([[1, 2, 3]]), np.array([[4], [5]])]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Three arrays with compatible shapes
    input_dict = {"args": [np.array([1, 2, 3]), np.array([[1], [2], [3]]), np.array(5)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Array with different data type
    input_dict = {"args": [np.array([1, 2, 3], dtype=np.int32), np.array([[1], [2], [3]], dtype=np.float64)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Arrays with negative values
    input_dict = {"args": [np.array([-1, -2, -3]), np.array([[-1], [-2], [-3]])]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D arrays
    input_dict = {"args": [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), np.array([1, 2])]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different dtypes with broadcasting
    input_dict = {"args": [np.array([1, 2, 3], dtype=np.float32), np.array(1, dtype=np.int32)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Broadcasting with shape (1, x) and (y, 1)
    input_dict = {"args": [np.array([[1, 2, 3]]), np.array([[4], [5]])]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.broadcast_arrays"] = tf_experimental_numpy_broadcast_arrays_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.broadcast_arrays' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.broadcast_arrays'.")

check_valid('tf.experimental.numpy.broadcast_arrays', generated_inputs['tf.experimental.numpy.broadcast_arrays'], lib="tf", suffix=0)
