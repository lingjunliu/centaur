
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

    # Input 3: List of floats
    input_dict = {"seq": [1.0, 2.0, 3.0]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: List of booleans
    input_dict = {"seq": [True, False, True]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: List of mixed types (but homogenous within nested lists)
    input_dict = {"seq": [1, True]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Nested list
    input_dict = {"seq": [[1, 2], [3, 4]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Deeply nested list
    input_dict = {"seq": [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: List containing a numpy array's first element
    input_dict = {"seq": [np.array([1, 2, 3]).tolist()[0]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: List with nested lists with same length
    input_dict = {"seq": [[1, 2], [3, 4]]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: List with integers
    input_dict = {"seq": [1, 2, 3, 4, 5]}
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
