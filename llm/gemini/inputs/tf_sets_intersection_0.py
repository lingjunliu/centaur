
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np
import collections

def tf_sets_intersection_inputs():
    list_of_inputs = []

    # Input 1: Basic case with dense tensors
    a = tf.constant([[1, 2, 3], [4, 5, 6]]).numpy()
    b = tf.constant([[2, 3, 4], [5, 6, 7]]).numpy()
    validate_indices = True
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shapes, but compatible
    a = tf.constant([[1, 2], [3, 4]]).numpy()
    b = tf.constant([[2, 1], [4, 3]]).numpy()
    validate_indices = False
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Dense with validate indices false
    a = tf.constant([[1, 2, 3], [4, 5, 6]]).numpy()
    b = tf.constant([[2, 3, 1], [5, 6, 4]]).numpy()
    validate_indices = False
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty intersection
    a = tf.constant([[1, 2], [3, 4]]).numpy()
    b = tf.constant([[5, 6], [7, 8]]).numpy()
    validate_indices = True
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: validate_indices=False
    a = tf.constant([[1, 2, 3], [4, 5, 6]]).numpy()
    b = tf.constant([[2, 3, 4], [5, 6, 7]]).numpy()
    validate_indices = False
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sets.intersection"] = tf_sets_intersection_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sets.intersection' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sets.intersection'.")

check_valid('tf.sets.intersection', generated_inputs['tf.sets.intersection'], lib="tf", suffix=0)
