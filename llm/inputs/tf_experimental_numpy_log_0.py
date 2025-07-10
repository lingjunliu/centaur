
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_log_inputs():
    list_of_inputs = []

    # Input 1: Positive scalar
    x = tf.constant(2.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive 1D array
    x = tf.constant([1.0, 2.0, 3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Positive 2D array
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive 3D array
    x = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Array with a very small positive value
    x = tf.constant([1e-8, 1.0, 2.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger values
    x = tf.constant([100.0, 1000.0, 10000.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float64 tensor
    x = tf.constant(np.array([1.0, 2.0], dtype=np.float64))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Small positive values
    x = tf.constant([0.1, 0.01, 0.001])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Multidimensional array with varying values
    x = tf.constant([[0.5, 1.5], [2.5, 3.5]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed large and small numbers
    x = tf.constant([0.0001, 1000.0, 2.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Tensor full of ones
    x = tf.ones((2, 3))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: tf.range
    x = tf.range(1, 6, dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.log"] = tf_experimental_numpy_log_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.log' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.log'.")

check_valid('tf.experimental.numpy.log', generated_inputs['tf.experimental.numpy.log'], lib="tf", suffix=0)
