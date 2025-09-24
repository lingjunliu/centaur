
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_strip_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array(["  TensorFlow  "])
    name = None
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array(["\nTensorFlow", "The python library    "])
    name = "strip_op"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array(["  ", " ", ""])
    name = None
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array(["  Leading", "Trailing  ", "  Both  ", "None"])
    name = "example_name"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([["  Nested  ", "  Array"], ["More   ", "Stuff "]])
    name = None
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array(" Single string with spaces ")
    name = "single_string"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array(["\tTab before", "Tab after\t", "\tBoth tabs\t"])
    name = None
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array(["\rCarriage return before", "Carriage return after\r", "\rBoth returns\r"])
    name = None
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array(["\n\t Mixed whitespace \r"])
    name = "mixed_whitespace"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array(["No whitespace"])
    name = None
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.strings.strip"] = tf_strings_strip_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.strip' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.strip'.")

check_valid('tf.strings.strip', generated_inputs['tf.strings.strip'], lib="tf", suffix=0)
