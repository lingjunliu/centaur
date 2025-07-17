
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ReadFile_inputs():
    list_of_inputs = []

    # Input 1: Valid filename
    filename = tf.constant(np.array("test_file_1.txt").astype(np.string_))
    name = "ReadFileOp1"
    input_dict = {"filename": filename, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Another valid filename
    filename = tf.constant(np.array("test_file_2.txt").astype(np.string_))
    name = "ReadFileOp2"
    input_dict = {"filename": filename, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Filename with a different path
    filename = tf.constant(np.array("./path/to/test_file_3.txt").astype(np.string_))
    name = "ReadFileOp3"
    input_dict = {"filename": filename, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4:  Filename as a numpy array
    filename = tf.constant(np.array("test_file_4.txt").astype(np.string_))
    name = "ReadFileOp4"
    input_dict = {"filename": filename, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Longer filename
    filename = tf.constant(np.array("a_very_long_filename_for_testing.txt").astype(np.string_))
    name = "ReadFileOp5"
    input_dict = {"filename": filename, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Filename with spaces
    filename = tf.constant(np.array("test file with spaces.txt").astype(np.string_))
    name = "ReadFileOp6"
    input_dict = {"filename": filename, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: Filename with special characters
    filename = tf.constant(np.array("test_file!@#$%.txt").astype(np.string_))
    name = "ReadFileOp7"
    input_dict = {"filename": filename, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8:  Filename with Unicode characters - REMOVED to avoid encoding errors

    # Input 9: Filename with numbers in name
    filename = tf.constant(np.array("test_file_123.txt").astype(np.string_))
    name = "ReadFileOp9"
    input_dict = {"filename": filename, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Filename with extension name
    filename = tf.constant(np.array("test_file.extension_name").astype(np.string_))
    name = "ReadFileOp10"
    input_dict = {"filename": filename, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ReadFile"] = tf_raw_ops_ReadFile_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ReadFile' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ReadFile'.")

check_valid('tf.raw_ops.ReadFile', generated_inputs['tf.raw_ops.ReadFile'], lib="tf", suffix=0)
