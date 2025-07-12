
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

    # Input 2: Scalar and vector
    a = np.array(5)
    b = np.array([1, 2, 3])
    input_dict = {"args": [tf.constant(a), tf.constant(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Vector and matrix
    a = np.array([1, 2, 3])
    b = np.array([[1], [2], [3]])
    input_dict = {"args": [tf.constant(a), tf.constant(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Two matrices with compatible shapes
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[7], [8]])
    input_dict = {"args": [tf.constant(a), tf.constant(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Three arrays with different dimensions
    a = np.array(1)
    b = np.array([1, 2, 3])
    c = np.array([[1], [2], [3]])
    input_dict = {"args": [tf.constant(a), tf.constant(b), tf.constant(c)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Two arrays with integer and float types
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {"args": [tf.constant(a), tf.constant(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7:  Arrays with different data types
    a = np.array([1, 2, 3])
    b = np.array([1.0, 2.0, 3.0])
    c = np.array([True, False, True])

    input_dict = {"args": [tf.constant(a), tf.constant(b), tf.constant(c)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Higher dimension tensors
    a = np.random.rand(2, 3, 4)
    b = np.random.rand(3, 4)
    input_dict = {"args": [tf.constant(a), tf.constant(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Examples with negative values
    a = np.array([-1, -2, -3])
    b = np.array([1, 2, 3])
    input_dict = {"args": [tf.constant(a), tf.constant(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Example with different shapes and types
    a = np.array([[1, 2], [3, 4]])
    b = np.array([5.0, 6.0])
    c = np.array(7)
    input_dict = {"args": [tf.constant(a), tf.constant(b), tf.constant(c)]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Example using tf.zeros
    a = tf.zeros((2,3))
    b = tf.constant(1.0)
    input_dict = {"args": [a, b]}
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
