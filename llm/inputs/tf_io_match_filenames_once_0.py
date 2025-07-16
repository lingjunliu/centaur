
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_match_filenames_once_inputs():
    list_of_inputs = []

    # Input 1
    pattern = tf.convert_to_tensor(np.array("*.txt").astype(np.str_), dtype=tf.string)
    name = "file_pattern_1"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    pattern = tf.convert_to_tensor(np.array("image*.png").astype(np.str_), dtype=tf.string)
    name = "image_pattern_2"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    pattern = tf.convert_to_tensor(np.array("data_*.csv").astype(np.str_), dtype=tf.string)
    name = "data_pattern_3"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    pattern = tf.convert_to_tensor(np.array("log_*.log").astype(np.str_), dtype=tf.string)
    name = "log_pattern_4"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    pattern = tf.convert_to_tensor(np.array("*").astype(np.str_), dtype=tf.string)
    name = "all_files_5"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    pattern = tf.convert_to_tensor(np.array("file?.dat").astype(np.str_), dtype=tf.string)
    name = "file_question_6"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    pattern = tf.convert_to_tensor(np.array("backup.*").astype(np.str_), dtype=tf.string)
    name = "backup_files_7"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: A tensor of file patterns
    pattern = tf.convert_to_tensor(np.array(["*.txt", "*.csv"]).astype(np.str_), dtype=tf.string)
    name = "multiple_patterns_8"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    pattern = tf.convert_to_tensor(np.array("results_*.out").astype(np.str_), dtype=tf.string)
    name = "results_9"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    pattern = tf.convert_to_tensor(np.array("temp_*.*").astype(np.str_), dtype=tf.string)
    name = "temp_files_10"
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
