
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_is_tensor_inputs():
    list_of_inputs = []

    # Input 1: Empty list
    input_dict = {"x": []}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List of integers
    input_dict = {"x": [1, 2, 3]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: List of floats
    input_dict = {"x": [1.0, 2.5, 3.7]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: List of booleans
    input_dict = {"x": [True, False, True]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: List of numpy arrays (scalar)
    input_dict = {"x": [np.array(1), np.array(2), np.array(3)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: List of numpy arrays (1D)
    input_dict = {"x": [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: List of numpy arrays (2D)
    input_dict = {"x": [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: List of tensorflow constants converted to numpy array
    input_dict = {"x": [tf.constant([1,2]).numpy(), tf.constant([3,4]).numpy()]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: List of numpy arrays with same shape
    input_dict = {"x": [np.array([1, 2]), np.array([3, 4]), np.array([5,6])]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: list of tf.Tensors
    input_dict = {"x": [tf.constant([1, 2]), tf.constant([3, 4]), tf.constant([5,6])]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.is_tensor"] = tf_is_tensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.is_tensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.is_tensor'.")

check_valid('tf.is_tensor', generated_inputs['tf.is_tensor'], lib="tf", suffix=0)
