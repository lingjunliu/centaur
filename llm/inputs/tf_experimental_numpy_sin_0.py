
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_sin_inputs():
    list_of_inputs = []

    # Input 1: Scalar
    x = tf.constant(0.0, dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array
    x = tf.constant(np.array([0, np.pi/2, np.pi]), dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array
    x = tf.constant(np.array([[0, np.pi/2], [np.pi, 3*np.pi/2]]), dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array
    x = tf.constant(np.random.rand(2, 3, 4), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values
    x = tf.constant(np.array([-np.pi/2, -np.pi]), dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large values
    x = tf.constant(np.array([100*np.pi, 200*np.pi]), dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Array with zeros
    x = tf.constant(np.array([0, 0, 0]), dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Mixed positive and negative
    x = tf.constant(np.array([-1, 0, 1, -2]), dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: High-dimensional array
    x = tf.constant(np.random.rand(2, 2, 2, 2), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float64
    x = tf.constant(np.array([0, np.pi/2, np.pi]), dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: scalar float64
    x = tf.constant(0.0, dtype=tf.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Scalar complex64
    x = tf.constant(1+1j, dtype=tf.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.sin"] = tf_experimental_numpy_sin_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.sin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.sin'.")

check_valid('tf.experimental.numpy.sin', generated_inputs['tf.experimental.numpy.sin'], lib="tf", suffix=0)
