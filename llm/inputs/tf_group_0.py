
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_group_inputs():
    list_of_inputs = []

    # Input 1: Empty list of tensors
    input_dict = {
        "inputs": [],
        "name": "group_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Single tensor
    a = tf.constant([1, 2, 3])
    input_dict = {
        "inputs": [a],
        "name": "group_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple tensors
    a = tf.constant([1, 2, 3])
    b = tf.constant([4, 5, 6])
    input_dict = {
        "inputs": [a, b],
        "name": "group_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.group"] = tf_group_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.group' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.group'.")

check_valid('tf.group', generated_inputs['tf.group'], lib="tf", suffix=0)
