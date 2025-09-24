
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_matching_files_inputs():
    list_of_inputs = []

    # Input 1: Basic wildcard
    pattern = "*.txt"
    name = None
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: More specific wildcard
    pattern = "data_*.csv"
    name = "my_files"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single file
    pattern = "myfile.pdf"
    name = None
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: No wildcard (just a directory)
    pattern = "mydir/"
    name = "dir_name"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multiple patterns (vector)
    pattern = ["*.jpg", "*.png"]
    name = None
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6:  Name with special characters
    pattern = "*.log"
    name = "log-files"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Path and wildcard
    pattern = "path/to/data/*.dat"
    name = None
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Numerical chars in pattern
    pattern = "file123*.txt"
    name = None
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Hidden files
    pattern = ".*"
    name = "hidden_files"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: empty name
    pattern = "*.tmp"
    name = ""
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.matching_files"] = tf_io_matching_files_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.matching_files' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.matching_files'.")

check_valid('tf.io.matching_files', generated_inputs['tf.io.matching_files'], lib="tf", suffix=0)
