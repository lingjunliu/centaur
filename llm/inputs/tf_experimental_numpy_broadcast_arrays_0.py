
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_broadcast_arrays_inputs():
    list_of_inputs = []

    # Input 1: Two scalars
    a = np.array(1)
    b = np.array(2)
    input_dict = {"args": [tf.constant(a), tf.constant(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar and 1D array
    a = np.array(5)
    b = np.array([1, 2, 3])
    input_dict = {"args": [tf.constant(a), tf.constant(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D and 2D arrays
    a = np.array([1, 2, 3])
    b = np.array([[1], [2], [3]])
    input_dict = {"args": [tf.constant(a), tf.constant(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Two 2D arrays, different shapes, broadcastable
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[7, 8, 9]])
    input_dict = {"args": [tf.constant(a), tf.constant(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D and 1D
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = np.array([1, 2])
    input_dict = {"args": [tf.constant(a), tf.constant(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D and 2D
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = np.array([[1, 2]])
    input_dict = {"args": [tf.constant(a), tf.constant(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different dtypes
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([4, 5, 6], dtype=np.float32)
    input_dict = {"args": [tf.constant(a), tf.constant(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D and 1D
    a = np.random.rand(2, 3, 4, 5)
    b = np.random.rand(5)
    input_dict = {"args": [tf.constant(a), tf.constant(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Multiple arrays
    a = np.array([1, 2, 3])
    b = np.array([[4], [5], [6]])
    c = np.array(7)
    input_dict = {"args": [tf.constant(a), tf.constant(b), tf.constant(c)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Two 3D arrays with compatible shapes
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = np.array([[[9, 10], [11, 12]]])
    input_dict = {"args": [tf.constant(a), tf.constant(b)]}
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
