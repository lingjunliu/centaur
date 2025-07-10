
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_broadcast_arrays_inputs():
    list_of_inputs = []

    # Input 1: Basic broadcast
    a = np.array([1, 2, 3])
    b = np.array([[1], [2], [3]])
    input_dict = {"args": [tf.constant(a), tf.constant(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar broadcast
    a = np.array(5)
    b = np.array([1, 2, 3])
    input_dict = {"args": [tf.constant(a), tf.constant(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: More complex broadcast
    a = np.array([[1, 2, 3]])
    b = np.array([[4], [5]])
    input_dict = {"args": [tf.constant(a), tf.constant(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcast with different dtypes
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    input_dict = {"args": [tf.constant(a), tf.constant(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcast with different shapes and dtypes
    a = np.array([1, 2], dtype=np.int64)
    b = np.array([[1.5, 2.5, 3.5]], dtype=np.float64)
    input_dict = {"args": [tf.constant(a), tf.constant(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting with 3 arrays
    a = np.array([1, 2, 3])
    b = np.array([[1], [2], [3]])
    c = np.array(5)
    input_dict = {"args": [tf.constant(a), tf.constant(b), tf.constant(c)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting with different dimensions
    a = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    input_dict = {"args": [tf.constant(a), tf.constant(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Broadcasting with negative values
    a = np.array([-1, -2, -3])
    b = np.array([[-4], [-5], [-6]])
    input_dict = {"args": [tf.constant(a), tf.constant(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Broadcasting with zeros
    a = np.array([0, 0, 0])
    b = np.array([[0], [0], [0]])
    input_dict = {"args": [tf.constant(a), tf.constant(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Broadcasting with complex numbers
    a = np.array([1+1j, 2+2j])
    b = np.array([[4+4j], [5+5j]])
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
