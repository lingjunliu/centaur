
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import inspect

def tf_raw_ops_invert_permutation_inputs():
    list_of_inputs = []

    # Input 1: Valid permutation
    x = np.array([3, 4, 0, 2, 1], dtype=np.int32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Another valid permutation with int64
    x = np.array([2, 0, 1], dtype=np.int64)
    input_dict = {"x": x, "name": "invert_perm_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: A longer permutation
    x = np.array([5, 0, 2, 1, 4, 3], dtype=np.int32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Simple ascending order (already inverted)
    x = np.array([0, 1, 2, 3], dtype=np.int64)
    input_dict = {"x": x, "name": "invert_perm_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Simple descending order
    x = np.array([3, 2, 1, 0], dtype=np.int32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Random permutation
    x = np.array([1, 0, 5, 4, 2, 3], dtype=np.int64)
    input_dict = {"x": x, "name": "invert_perm_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Another random permutation
    x = np.array([4, 2, 0, 1, 3], dtype=np.int32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small permutation
    x = np.array([1, 0], dtype=np.int64)
    input_dict = {"x": x, "name": "invert_perm_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: A longer permutation int32
    x = np.array([7, 3, 0, 5, 4, 1, 6, 2], dtype=np.int32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single element permutation
    x = np.array([0], dtype=np.int64)
    input_dict = {"x": x, "name": "invert_perm_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.InvertPermutation"] = tf_raw_ops_invert_permutation_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.InvertPermutation' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.InvertPermutation'.")

check_valid('tf.raw_ops.InvertPermutation', generated_inputs['tf.raw_ops.InvertPermutation'], lib="tf", suffix=0)
