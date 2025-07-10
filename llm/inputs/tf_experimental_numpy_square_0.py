
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_square_inputs():
    list_of_inputs = []

    # Input 1: Scalar
    x = tf.constant(5)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array with positive values
    x = tf.constant([1, 2, 3, 4, 5])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array with negative values
    x = tf.constant([-1, -2, -3, -4, -5])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array with mixed positive and negative values
    x = tf.constant([-1, 2, -3, 4, -5])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array
    x = tf.constant([[1, 2], [3, 4]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Array with float values
    x = tf.constant([1.5, 2.5, 3.5])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Array with mixed float and integer values
    x = tf.constant([1, 2.5, -3, -4.5])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zero tensor
    x = tf.constant([0, 0, 0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single element tensor
    x = tf.constant([7])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.square"] = tf_experimental_numpy_square_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.square' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.square'.")

check_valid('tf.experimental.numpy.square', generated_inputs['tf.experimental.numpy.square'], lib="tf", suffix=0)
