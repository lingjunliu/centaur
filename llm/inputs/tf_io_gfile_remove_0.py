
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import os

def tf_io_gfile_remove_inputs():
    list_of_inputs = []

    # Create dummy files to ensure they exist
    dummy_files = [
        "test_file.txt",
        "test_file_" + "a" * 200 + ".txt",
        "test_file!@#$%^&*.txt",
        "测试文件.txt",
        "test file.txt",
        "abc.txt",
        "123.txt",
        "xyz.txt",
        "uvw.txt",
        "rst.txt"
    ]

    for file in dummy_files:
        try:
            with open(file, "w") as f:
                f.write("dummy content")
        except Exception as e:
            print(f"Failed to create dummy file {file}: {e}")
            pass

    # Input 1: Simple valid path
    path = "test_file.txt"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Path with a long filename
    path = "test_file_" + "a" * 200 + ".txt"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Path with special characters in filename
    path = "test_file!@#$%^&*.txt"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Path with unicode characters
    path = "测试文件.txt"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Path with a space in the filename
    path = "test file.txt"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Simple valid path
    path = "abc.txt"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Simple valid path
    path = "123.txt"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Simple valid path
    path = "xyz.txt"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Simple valid path
    path = "uvw.txt"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Simple valid path
    path = "rst.txt"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.gfile.remove"] = tf_io_gfile_remove_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.gfile.remove' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.gfile.remove'.")

check_valid('tf.io.gfile.remove', generated_inputs['tf.io.gfile.remove'], lib="tf", suffix=0)
