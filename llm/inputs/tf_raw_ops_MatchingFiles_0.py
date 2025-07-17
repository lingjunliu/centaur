
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_matching_files_inputs():
    list_of_inputs = []

    # Input 1: Basic glob pattern
    pattern = np.array("./*", dtype=np.string_)
    name = "matching_files_1"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: More specific glob pattern
    pattern = np.array("./*.txt", dtype=np.string_)
    name = "matching_files_2"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple patterns
    pattern = np.array(["./*.txt", "./*.py"], dtype=np.string_)
    name = "matching_files_3"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Pattern with a character class
    pattern = np.array("./file[0-9].txt", dtype=np.string_)
    name = "matching_files_4"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Pattern with multiple wildcards
    pattern = np.array("./*.*", dtype=np.string_)
    name = "matching_files_5"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Pattern for a specific file
    pattern = np.array("./myfile.txt", dtype=np.string_)
    name = "matching_files_6"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7:  Empty pattern
    pattern = np.array("", dtype=np.string_)
    name = "matching_files_7"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Pattern with a question mark
    pattern = np.array("./file?.txt", dtype=np.string_)
    name = "matching_files_8"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Vector pattern with some empty strings
    pattern = np.array(["./*.txt", ""], dtype=np.string_)
    name = "matching_files_9"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: A longer name
    pattern = np.array("./*.log", dtype=np.string_)
    name = "a_very_long_name_for_matching_files_operation"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MatchingFiles"] = tf_raw_ops_matching_files_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MatchingFiles' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatchingFiles'.")

check_valid('tf.raw_ops.MatchingFiles', generated_inputs['tf.raw_ops.MatchingFiles'], lib="tf", suffix=0)
