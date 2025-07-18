
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

# NOTE: A conflict exists between the testing framework and the TensorFlow API.
# 1. The framework's signature check requires a 'name' key, causing a KeyError if it's omitted.
# 2. Passing a string value for 'name' to tf.case in the framework's eager execution
#    environment causes an `AttributeError: Tensor.name is undefined`.
# This solution attempts to resolve this conflict by setting `name=None`.
# The expectation is that this satisfies the framework's key existence check,
# while being handled by the TensorFlow API as if the optional argument was not
# provided, thus avoiding the AttributeError.
# Placeholders are still used for callable arguments due to other framework limitations.

def tf_case_inputs():
    list_of_inputs = []

    # Input 1: Using name=None to satisfy framework while avoiding TF eager error.
    input_dict_1 = {
        'pred_fn_pairs': [(tf.constant(True, dtype=tf.bool), 1)],
        'default': [2],
        'exclusive': False,
        'strict': False,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Default branch taken with name=None
    input_dict_2 = {
        'pred_fn_pairs': [(tf.constant(False, dtype=tf.bool), 1)],
        'default': [2],
        'exclusive': False,
        'strict': False,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Exclusive=True, second branch taken
    input_dict_3 = {
        'pred_fn_pairs': [
            (tf.constant(False, dtype=tf.bool), 1),
            (tf.constant(True, dtype=tf.bool), 2)
        ],
        'default': [-1],
        'exclusive': True,
        'strict': False,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Exclusive=False, first of two true predicates is taken
    input_dict_4 = {
        'pred_fn_pairs': [
            (tf.constant(True, dtype=tf.bool), 1),
            (tf.constant(True, dtype=tf.bool), 2)
        ],
        'default': [-1],
        'exclusive': False,
        'strict': False,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Multiple pairs, none true, default is taken
    input_dict_5 = {
        'pred_fn_pairs': [
            (tf.constant(False, dtype=tf.bool), 1),
            (tf.constant(False, dtype=tf.bool), 2)
        ],
        'default': [3],
        'exclusive': False,
        'strict': False,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Strict=True case
    input_dict_6 = {
        'pred_fn_pairs': [
            (tf.constant(True, dtype=tf.bool), 1)
        ],
        'default': [0],
        'exclusive': False,
        'strict': True,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Empty list of pairs, always uses default
    input_dict_7 = {
        'pred_fn_pairs': [],
        'default': [42],
        'exclusive': False,
        'strict': False,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Another exclusive case with multiple false predicates
    input_dict_8 = {
        'pred_fn_pairs': [
            (tf.constant(False, dtype=tf.bool), 10),
            (tf.constant(False, dtype=tf.bool), 20),
            (tf.constant(True, dtype=tf.bool), 30)
        ],
        'default': [0],
        'exclusive': True,
        'strict': False,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: More pairs for non-exclusive with negative placeholder values
    input_dict_9 = {
        'pred_fn_pairs': [
            (tf.constant(False, dtype=tf.bool), -1),
            (tf.constant(True, dtype=tf.bool), -2),
            (tf.constant(True, dtype=tf.bool), -3)
        ],
        'default': [-99],
        'exclusive': False,
        'strict': False,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Simple case with exclusive=True
    input_dict_10 = {
        'pred_fn_pairs': [(tf.constant(True, dtype=tf.bool), 100)],
        'default': [200],
        'exclusive': True,
        'strict': False,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.case"] = tf_case_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.case' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.case'.")

check_valid('tf.case', generated_inputs['tf.case'], lib="tf", suffix=0)
