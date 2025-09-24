
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np
import os

def tf_io_gfile_isdir_inputs():
    list_of_inputs = []

    # Input 1: Empty string
    input_dict = {"path": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Current directory
    input_dict = {"path": "."}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Non-existent directory
    input_dict = {"path": "non_existent_directory"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Existing directory
    existing_dir = "test_dir"
    if not os.path.exists(existing_dir):
      os.makedirs(existing_dir)
    input_dict = {"path": existing_dir}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Existing file
    existing_file = "test_file.txt"
    with open(existing_file, "w") as f:
      f.write("test")
    input_dict = {"path": existing_file}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Relative path
    input_dict = {"path": "./test_dir"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Absolute path
    abs_path = os.path.abspath(existing_dir)
    input_dict = {"path": abs_path}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Path with spaces
    dir_with_spaces = "test dir with spaces"
    if not os.path.exists(dir_with_spaces):
        try:
            os.makedirs(dir_with_spaces)
        except OSError:
            dir_with_spaces = "testdirwithspaces"  # Fallback if spaces cause issues on certain systems
            if not os.path.exists(dir_with_spaces):
                os.makedirs(dir_with_spaces)
    
    input_dict = {"path": dir_with_spaces}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    # Input 9: Path with special characters
    special_dir = "test!@#$%^&*()"
    if not os.path.exists(special_dir):
      try:
        os.makedirs(special_dir)
      except OSError:
        special_dir = "test" # Fallback if special chars cause issues
        if not os.path.exists(special_dir):
           os.makedirs(special_dir)
    input_dict = {"path": special_dir}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Long path
    long_path = "a" * 256
    input_dict = {"path": long_path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.gfile.isdir"] = tf_io_gfile_isdir_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.gfile.isdir' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.gfile.isdir'.")

check_valid('tf.io.gfile.isdir', generated_inputs['tf.io.gfile.isdir'], lib="tf", suffix=0)
