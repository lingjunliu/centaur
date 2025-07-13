
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_identity_n_inputs():
    list_of_inputs = []

    # Input 1: Empty list
    input_list = []
    input_dict = {"input": input_list, "name": "empty_list"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List with one tensor
    input_list = [tf.constant(np.array([1, 2, 3]))]
    input_dict = {"input": input_list, "name": "single_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: List with multiple tensors of same shapes
    input_list = [tf.constant(np.array([1, 2])), tf.constant(np.array([3, 4]))]
    input_dict = {"input": input_list, "name": "same_shapes"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: List with multiple tensors of different types, but convertible
    input_list = [tf.constant(np.array([1, 2], dtype=np.int32)), tf.constant(np.array([3, 4], dtype=np.int64))]
    input_dict = {"input": input_list, "name": "diff_types_convertible"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: List with 2D tensors
    input_list = [tf.constant(np.random.rand(2, 3))]
    input_dict = {"input": input_list, "name": "2d_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: List with complex numbers
    input_list = [tf.constant(np.array([1 + 1j, 2 + 2j]))]
    input_dict = {"input": input_list, "name": "complex_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: List with boolean tensors
    input_list = [tf.constant(np.array([True, False, True]))]
    input_dict = {"input": input_list, "name": "bool_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multiple tensors with a specified name
    input_list = [tf.constant(np.array([1, 2, 3])), tf.constant(np.array([4, 5, 6]))]
    input_dict = {"input": input_list, "name": "named_tensors"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: List with negative values
    input_list = [tf.constant(np.array([-1, -2, -3]))]
    input_dict = {"input": input_list, "name": "negative_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Small tensors
    input_list = [tf.constant(np.random.rand(5, 5))]
    input_dict = {"input": input_list, "name": "small_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.identity_n"] = tf_identity_n_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.identity_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.identity_n'.")

check_valid('tf.identity_n', generated_inputs['tf.identity_n'], lib="tf", suffix=0)
