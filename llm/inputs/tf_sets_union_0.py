
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sets_union_inputs():
    list_of_inputs = []

    # Input 1: 2D tensors, compatible shapes
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[3, 4], [5, 6]])
    validate_indices = False
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensors, different sizes in last dimension
    a = tf.constant([[1, 2, 3], [4, 5, 6]])
    b = tf.constant([[4, 5], [6, 7]])
    validate_indices = False
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tensors with duplicate values
    a = tf.constant([[1, 2, 2], [3, 4, 4]])
    b = tf.constant([[2, 3, 3], [4, 5, 5]])
    validate_indices = True
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: a and b are equal
    a = tf.constant([[1, 2, 3], [4, 5, 6]])
    b = tf.constant([[1, 2, 3], [4, 5, 6]])
    validate_indices = True
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int64
    a = tf.constant([[1, 2], [3, 4]], dtype=tf.int64)
    b = tf.constant([[3, 4], [5, 6]], dtype=tf.int64)
    validate_indices = False
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: validate indices true
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[3, 4], [5, 6]])
    validate_indices = True
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Convert to numpy arrays
    for input_dict in list_of_inputs:
        input_dict["a"] = input_dict["a"].numpy()
        input_dict["b"] = input_dict["b"].numpy()

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sets.union"] = tf_sets_union_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sets.union' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sets.union'.")

check_valid('tf.sets.union', generated_inputs['tf.sets.union'], lib="tf", suffix=0)
