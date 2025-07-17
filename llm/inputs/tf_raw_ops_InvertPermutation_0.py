
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_invert_permutation_inputs():
    list_of_inputs = []

    # Input 1: Basic example
    x = np.array([3, 4, 0, 2, 1], dtype=np.int32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different permutation
    x = np.array([0, 1, 2, 3], dtype=np.int32)
    input_dict = {"x": x, "name": "permutation"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Larger permutation
    x = np.array([5, 0, 2, 1, 3, 4], dtype=np.int32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4:  int64 type
    x = np.array([3, 4, 0, 2, 1], dtype=np.int64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different order
    x = np.array([4, 2, 3, 0, 1], dtype=np.int32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: simple permutation
    x = np.array([1, 0], dtype=np.int32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: More complex permutation
    x = np.array([6, 1, 3, 4, 0, 2, 5], dtype=np.int32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger int64
    x = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Another example with int32
    x = np.array([7, 5, 3, 1, 6, 0, 2, 4], dtype=np.int32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10: Example with a different name
    x = np.array([2, 0, 1], dtype=np.int32)
    input_dict = {"x": x, "name": "test_invert"}
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
