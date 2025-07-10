
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import os

def tf_io_gfile_stat_inputs():
    list_of_inputs = []

    # Input 1: A valid file path
    file_path = "temp_file1.txt"
    with open(file_path, "w") as f:
        f.write("Test content")
    input_dict = {"path": file_path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Another valid file path
    file_path = "temp_file2.txt"
    with open(file_path, "w") as f:
        f.write("Another test content")
    input_dict = {"path": file_path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty file
    file_path = "temp_file3.txt"
    open(file_path, 'a').close()
    input_dict = {"path": file_path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: File path with spaces
    file_path = "temp file4.txt"
    with open(file_path, "w") as f:
        f.write("Content with spaces")
    input_dict = {"path": file_path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: File path with special characters
    file_path = "temp_file5!.txt"
    with open(file_path, "w") as f:
        f.write("Content with special characters")
    input_dict = {"path": file_path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Relative path
    file_path = "./temp_file6.txt"
    with open(file_path, "w") as f:
        f.write("Relative path content")
    input_dict = {"path": file_path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Path to existing directory
    dir_path = "."
    input_dict = {"path": dir_path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.gfile.stat"] = tf_io_gfile_stat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.gfile.stat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.gfile.stat'.")

check_valid('tf.io.gfile.stat', generated_inputs['tf.io.gfile.stat'], lib="tf", suffix=0)
