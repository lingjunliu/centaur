
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_gfile_makedirs_inputs():
    list_of_inputs = []

    # Input 1: Simple relative path
    input_dict = {"path": "test_dir"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Nested relative path
    input_dict = {"path": "test_dir/nested_dir"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Absolute path (assuming a valid absolute path for the system)
    input_dict = {"path": "/tmp/absolute_dir"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Path with multiple nested directories
    input_dict = {"path": "dir1/dir2/dir3/dir4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Path with special characters in directory name
    input_dict = {"path": "special_chars!@#$%^&*()_+=-`~/dir"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Path with spaces in directory name
    input_dict = {"path": "dir with spaces"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Deeper nesting with special characters
    input_dict = {"path": "a/b/c!@#/d/e"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Path with a combination of spaces and special characters
    input_dict = {"path": "dir with space and !@#"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Very long path.
    input_dict = {"path": "a/" * 50 + "long_dir"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Path that overlaps with existing paths in other tests (should still work because makedirs succeeds if the directory already exists)
    input_dict = {"path": "test_dir/nested_dir/overlap_dir"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.gfile.makedirs"] = tf_io_gfile_makedirs_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.gfile.makedirs' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.gfile.makedirs'.")

check_valid('tf.io.gfile.makedirs', generated_inputs['tf.io.gfile.makedirs'], lib="tf", suffix=0)
