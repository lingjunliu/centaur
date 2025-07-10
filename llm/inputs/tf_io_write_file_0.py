
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_write_file_inputs():
    list_of_inputs = []

    # Input 1: Basic usage
    filename = np.array("test_file_1.txt").astype(np.string_)
    contents = np.array("Hello, world!").astype(np.string_)
    name = None

    input_dict = {
        "filename": filename,
        "contents": contents,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different filename
    filename = np.array("another_file.txt").astype(np.string_)
    contents = np.array("This is another file.").astype(np.string_)
    name = "write_operation"

    input_dict = {
        "filename": filename,
        "contents": contents,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty contents
    filename = np.array("empty_file.txt").astype(np.string_)
    contents = np.array("").astype(np.string_)
    name = None

    input_dict = {
        "filename": filename,
        "contents": contents,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Long contents
    filename = np.array("long_file.txt").astype(np.string_)
    contents = np.array("This is a very long string that should be written to the file. " * 100).astype(np.string_)
    name = "long_write"

    input_dict = {
        "filename": filename,
        "contents": contents,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5: Contents with special characters
    filename = np.array("special_chars.txt").astype(np.string_)
    contents = np.array("!@#$%^&*()_+=-`~[]{}|;':\",./<>?").astype(np.string_)
    name = None

    input_dict = {
        "filename": filename,
        "contents": contents,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Filename with directory path
    filename = np.array("path/to/new/file.txt").astype(np.string_)
    contents = np.array("This file is in a subdirectory.").astype(np.string_)
    name = "path_write"

    input_dict = {
        "filename": filename,
        "contents": contents,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Numerical content as string
    filename = np.array("numerical_file.txt").astype(np.string_)
    contents = np.array("1234567890").astype(np.string_)
    name = "numerical_write"

    input_dict = {
        "filename": filename,
        "contents": contents,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Filename with spaces
    filename = np.array("file with spaces.txt").astype(np.string_)
    contents = np.array("Content for file with spaces").astype(np.string_)
    name = None

    input_dict = {
        "filename": filename,
        "contents": contents,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large file content
    filename = np.array("large_file.txt").astype(np.string_)
    contents = np.array("A" * (2 * 1024 * 1024)).astype(np.string_)
    name = "large_write"

    input_dict = {
        "filename": filename,
        "contents": contents,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Simple file
    filename = np.array("simple_file.txt").astype(np.string_)
    contents = np.array("Simple content").astype(np.string_)
    name = "simple_write"

    input_dict = {
        "filename": filename,
        "contents": contents,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.write_file"] = tf_io_write_file_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.write_file' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.write_file'.")

check_valid('tf.io.write_file', generated_inputs['tf.io.write_file'], lib="tf", suffix=0)
