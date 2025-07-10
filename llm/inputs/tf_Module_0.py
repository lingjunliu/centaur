
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_module_inputs():
    list_of_inputs = []

    # Input 1: Simple name
    input_dict = {"name": "my_module"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Name with numbers
    input_dict = {"name": "module_123"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Name with underscores
    input_dict = {"name": "my_module_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Name with capitalization
    input_dict = {"name": "MyModuleName"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Longer name
    input_dict = {"name": "a_very_long_module_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Name starting with underscore
    input_dict = {"name": "_private_module"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Name ending with number
    input_dict = {"name": "module1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Name with numbers in the middle
    input_dict = {"name": "module123name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Name with all small letters
    input_dict = {"name": "smallmodule"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: A more descriptive name
    input_dict = {"name": "descriptive_module_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.Module"] = tf_module_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.Module' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.Module'.")

check_valid('tf.Module', generated_inputs['tf.Module'], lib="tf", suffix=0)
