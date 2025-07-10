
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_add_n_inputs():
    list_of_inputs = []

    # Input 1: Basic test with integers
    inputs = [tf.constant(np.array([[1, 2], [3, 4]])), tf.constant(np.array([[5, 6], [7, 8]]))]
    name = "add_n_test1"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Test with negative integers
    inputs = [tf.constant(np.array([[-1, 2], [-3, 4]])), tf.constant(np.array([[5, -6], [7, -8]]))]
    name = "add_n_test2"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Test with floats
    inputs = [tf.constant(np.array([[1.5, 2.5], [3.5, 4.5]])), tf.constant(np.array([[5.5, 6.5], [7.5, 8.5]]))]
    name = "add_n_test3"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Test with 3 tensors
    inputs = [tf.constant(np.array([[1, 2], [3, 4]])), tf.constant(np.array([[5, 6], [7, 8]])), tf.constant(np.array([[9, 10], [11, 12]]))]
    name = "add_n_test5"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Test with 1D tensors
    inputs = [tf.constant(np.array([1, 2, 3])), tf.constant(np.array([4, 5, 6]))]
    name = "add_n_test6"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.add_n"] = tf_math_add_n_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.add_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.add_n'.")

check_valid('tf.math.add_n', generated_inputs['tf.math.add_n'], lib="tf", suffix=0)
