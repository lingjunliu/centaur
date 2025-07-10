
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_invert_permutation_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    x = np.array([3, 4, 0, 2, 1], dtype=np.int32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different permutation
    x = np.array([2, 0, 1], dtype=np.int32)
    name = "perm2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Longer permutation
    x = np.array([5, 0, 1, 7, 8, 2, 4, 3, 6], dtype=np.int64)
    name = "perm3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Simple reverse order
    x = np.array([4, 3, 2, 1, 0], dtype=np.int32)
    name = "perm4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Identity permutation
    x = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    name = "perm5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Int64 data type
    x = np.array([1, 0], dtype=np.int64)
    name = "perm6"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Another Int64 example
    x = np.array([6, 7, 8, 9, 3, 4, 5, 0, 1, 2], dtype=np.int64)
    name = "perm7"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different name
    x = np.array([2, 1, 0], dtype=np.int32)
    name = "my_inverse"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger numbers, still valid permutation
    x = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10: Another valid permutation
    x = np.array([0, 2, 4, 1, 3], dtype=np.int32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.invert_permutation"] = tf_math_invert_permutation_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.invert_permutation' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.invert_permutation'.")

check_valid('tf.math.invert_permutation', generated_inputs['tf.math.invert_permutation'], lib="tf", suffix=0)
