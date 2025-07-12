
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nest_is_nested_inputs():
    list_of_inputs = []

    # Input 1: Empty list
    input_dict = {"seq": []}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List of integers
    input_dict = {"seq": [1, 2, 3]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: List of lists
    input_dict = {"seq": [[1, 2], [3, 4]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: List of numpy arrays (homogeneous shape)
    input_dict = {"seq": [np.array([1, 2]), np.array([3, 4])]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: List of mixed numeric types.
    input_dict = {"seq": [1, 2.5, 3]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Deeply nested list (flattened a little to avoid min/max issues)
    input_dict = {"seq": [[1, 2, 3, 4], 5]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: List containing None
    input_dict = {"seq": [1, None, 3]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: List of lists of numpy arrays (homogeneous shapes)
    input_dict = {"seq": [[np.array([1, 2]), np.array([3, 4])], [np.array([5, 6])]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: List of empty lists
    input_dict = {"seq": [[], [], []]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: List containing boolean values
    input_dict = {"seq": [True, False, True]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nest.is_nested"] = tf_nest_is_nested_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nest.is_nested' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nest.is_nested'.")

check_valid('tf.nest.is_nested', generated_inputs['tf.nest.is_nested'], lib="tf", suffix=0)
