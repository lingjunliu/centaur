
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
    input_dict = {"args": [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar and vector
    a = np.array(1)
    b = np.array([1, 2, 3])
    input_dict = {"args": [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Vector and matrix
    a = np.array([1, 2, 3])
    b = np.array([[1, 2, 3], [4, 5, 6]])
    input_dict = {"args": [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Two vectors of different sizes (broadcastable)
    a = np.array([1, 2, 3])
    b = np.array([4])
    input_dict = {"args": [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Multiple arrays with different shapes
    a = np.array(1)
    b = np.array([1, 2, 3])
    c = np.array([[1], [2], [3]])
    input_dict = {"args": [tf.convert_to_tensor(a), tf.convert_to_tensor(b), tf.convert_to_tensor(c)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Two matrices with compatible shapes
    a = np.array([[1, 2, 3]])
    b = np.array([[4], [5]])
    input_dict = {"args": [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Higher dimensional arrays
    a = np.random.rand(2, 3, 4)
    b = np.random.rand(3, 4)
    input_dict = {"args": [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: One array
    a = np.array([1, 2, 3])
    input_dict = {"args": [tf.convert_to_tensor(a)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Array with different datatype
    a = np.array([1.0, 2.0, 3.0])
    b = np.array(2.0)
    input_dict = {"args": [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array with int64
    a = np.array([1, 2, 3], dtype=np.int64)
    b = np.array(2, dtype=np.int64)
    input_dict = {"args": [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Negative values
    a = np.array([-1, 2, -3])
    b = np.array([-2])
    input_dict = {"args": [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]}
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
