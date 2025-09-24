
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_broadcast_to_inputs():
    list_of_inputs = []

    # Input 1: Basic example
    array = tf.constant(np.array([1, 2, 3]))
    shape = (3, 3)
    input_dict = {"array": array, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Broadcasting a scalar
    array = tf.constant(np.array(5))
    shape = (2, 2)
    input_dict = {"array": array, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Broadcasting a 2D array
    array = tf.constant(np.array([[1, 2], [3, 4]]))
    shape = (2, 2)
    input_dict = {"array": array, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting along a different axis
    array = tf.constant(np.array([[1], [2], [3]]))
    shape = (3, 4)
    input_dict = {"array": array, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting with higher dimensions
    array = tf.constant(np.array([1]))
    shape = (2, 3, 4)
    input_dict = {"array": array, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: More complex broadcasting
    array = tf.constant(np.array([[1, 2, 3]]))
    shape = (5, 3)
    input_dict = {"array": array, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting a 3D array
    array = tf.constant(np.array([[[1], [2]], [[3], [4]]]))
    shape = (2, 2, 5)
    input_dict = {"array": array, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Broadcasting a 1D array to a higher dimension
    array = tf.constant(np.array([1, 2]))
    shape = (2, 2, 2)
    input_dict = {"array": array, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different shaped 2D array
    array = tf.constant(np.array([[1, 2], [3, 4], [5, 6]]))
    shape = (3, 2)
    input_dict = {"array": array, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Broadcasting a 1D array to a larger shape
    array = tf.constant(np.array([1]))
    shape = (5, 5)
    input_dict = {"array": array, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.broadcast_to"] = tf_experimental_numpy_broadcast_to_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.broadcast_to' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.broadcast_to'.")

check_valid('tf.experimental.numpy.broadcast_to', generated_inputs['tf.experimental.numpy.broadcast_to'], lib="tf", suffix=0)
