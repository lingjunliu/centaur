
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_match_filenames_once_inputs():
    list_of_inputs = []

    # Input 1
    pattern = np.array("*.txt", dtype=np.str_)
    name = "match_files_1"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    pattern = np.array("file?.txt", dtype=np.str_)
    name = "match_files_2"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    pattern = np.array("data/file*.csv", dtype=np.str_)
    name = "match_files_3"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: multiple patterns
    pattern = np.array(["*.txt", "*.csv"], dtype=np.str_)
    name = "match_files_4"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: empty pattern
    pattern = np.array("", dtype=np.str_)
    name = "match_files_5"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex pattern
    pattern = np.array("**/file*.log", dtype=np.str_)
    name = "match_files_6"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: pattern with numbers
    pattern = np.array("data_123.dat", dtype=np.str_)
    name = "match_files_7"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: pattern with special characters
    pattern = np.array("file[1-5].txt", dtype=np.str_)
    name = "match_files_8"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: very specific file name
    pattern = np.array("exact_file_name.pdf", dtype=np.str_)
    name = "match_files_9"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: relative path
    pattern = np.array("./*.py", dtype=np.str_)
    name = "match_files_10"
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
