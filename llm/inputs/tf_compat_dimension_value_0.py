
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_compat_dimension_value_inputs():
    list_of_inputs = []

    # Input 1: Positive integer
    input_dict = {"dimension": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Zero
    input_dict = {"dimension": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative integer
    input_dict = {"dimension": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger positive integer
    input_dict = {"dimension": 1000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger negative integer
    input_dict = {"dimension": -1000}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: A small positive integer
    input_dict = {"dimension": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: A small negative integer
    input_dict = {"dimension": -5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: A prime number
    input_dict = {"dimension": 7}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Another positive integer
    input_dict = {"dimension": 23}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another negative integer
    input_dict = {"dimension": -42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.compat.dimension_value"] = tf_compat_dimension_value_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.compat.dimension_value' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.compat.dimension_value'.")

check_valid('tf.compat.dimension_value', generated_inputs['tf.compat.dimension_value'], lib="tf", suffix=0)
