
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_match_filenames_once_inputs():
    list_of_inputs = []

    # Input 1: Simple glob pattern
    pattern = tf.constant("*.txt", dtype=tf.string)
    name = "txt_files"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different glob pattern
    pattern = tf.constant("data?.csv", dtype=tf.string)
    name = "csv_files"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Pattern with directory
    pattern = tf.constant("mydir/*.jpg", dtype=tf.string)
    name = "jpg_files"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: More specific pattern
    pattern = tf.constant("log_2023-12-??.txt", dtype=tf.string)
    name = "log_files"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Pattern with subdirectories
    pattern = tf.constant("path/to/data/*", dtype=tf.string)
    name = "data_files"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: No name
    pattern = tf.constant("*.py", dtype=tf.string)
    name = None
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty pattern
    pattern = tf.constant("", dtype=tf.string)
    name = "empty_files"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different name
    pattern = tf.constant("*", dtype=tf.string)
    name = "all_files"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Multiple patterns
    pattern = tf.constant(["*.txt", "*.csv"], dtype=tf.string)
    name = "txt_csv_files"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Pattern with special characters
    pattern = tf.constant("file[0-9]*.dat", dtype=tf.string)
    name = "dat_files"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.match_filenames_once"] = tf_io_match_filenames_once_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.match_filenames_once' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.match_filenames_once'.")

check_valid('tf.io.match_filenames_once', generated_inputs['tf.io.match_filenames_once'], lib="tf", suffix=0)
