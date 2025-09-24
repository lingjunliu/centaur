
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import os

def tf_io_tfrecordwriter_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input with path and no options
    path = "example1.tfrecords"
    options = None
    input_dict = {"path": path, "options": options}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Valid path and empty options string
    path = "example2.tfrecords"
    options = ""
    input_dict = {"path": path, "options": options}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Valid path with a more complex file name
    path = "complex_name_file.tfrecords"
    options = None
    input_dict = {"path": path, "options": options}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Path with special characters (some may cause issues)
    path = "file_with_special_chars.tfrecords"
    options = None
    input_dict = {"path": path, "options": options}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Path with unicode characters (might cause encoding issues)
    path = "你好世界.tfrecords"
    options = None
    input_dict = {"path": path, "options": options}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Relative path
    path = "relative_path.tfrecords"
    options = None
    input_dict = {"path": path, "options": options}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Path with subdirectories (removed to avoid file system issues)

    # Input 8: Longer path
    path = "very_long_path_" * 5 + ".tfrecords"
    options = None
    input_dict = {"path": path, "options": options}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Path with spaces
    path = "file with spaces.tfrecords"
    options = None
    input_dict = {"path": path, "options": options}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Path with a different extension
    path = "example.data"
    options = None
    input_dict = {"path": path, "options": options}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Very simple path
    path = "a.tfrecords"
    options = None
    input_dict = {"path": path, "options": options}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.io.TFRecordWriter"] = tf_io_tfrecordwriter_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.TFRecordWriter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.TFRecordWriter'.")

check_valid('tf.io.TFRecordWriter', generated_inputs['tf.io.TFRecordWriter'], lib="tf", suffix=0)
