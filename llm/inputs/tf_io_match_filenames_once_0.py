
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_match_filenames_once_inputs():
    list_of_inputs = []

    # Input 1: Simple pattern
    pattern = tf.convert_to_tensor(np.array("*.txt", dtype=np.object), dtype=tf.string)
    name = "txt_files"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Pattern with directory
    pattern = tf.convert_to_tensor(np.array("data/*.csv", dtype=np.object), dtype=tf.string)
    name = "csv_files"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple patterns
    pattern = tf.convert_to_tensor(np.array(["*.txt", "*.csv"], dtype=np.object), dtype=tf.string)
    name = "text_and_csv"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: More complex pattern with wildcards
    pattern = tf.convert_to_tensor(np.array("data/*/*.log", dtype=np.object), dtype=tf.string)
    name = "nested_logs"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Using a different name
    pattern = tf.convert_to_tensor(np.array("*.py", dtype=np.object), dtype=tf.string)
    name = "python_scripts"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty pattern - Might not match anything but is valid
    pattern = tf.convert_to_tensor(np.array("", dtype=np.object), dtype=tf.string)
    name = "empty_pattern"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Pattern with special characters
    pattern = tf.convert_to_tensor(np.array("data-[0-9]*.dat", dtype=np.object), dtype=tf.string)
    name = "data_files"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Absolute path pattern
    pattern = tf.convert_to_tensor(np.array("/tmp/*.tmp", dtype=np.object), dtype=tf.string)
    name = "tmp_files"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Unicode characters in the pattern
    pattern = tf.convert_to_tensor(np.array("数据/*.txt", dtype=np.object), dtype=tf.string)
    name = "unicode_files"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another variation of name
    pattern = tf.convert_to_tensor(np.array("*.md", dtype=np.object), dtype=tf.string)
    name = "markdown_files_v2"
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
