
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import os

def tf_io_gfile_exists_inputs():
    list_of_inputs = []

    # Input 1: Valid path to an existing file
    with open("/tmp/test_file.txt", "w") as f:
        f.write("This is a test file.")
    input_dict = {"path": "/tmp/test_file.txt"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    os.remove("/tmp/test_file.txt")

    # Input 2: Valid path to an existing directory
    os.makedirs("/tmp/test_dir", exist_ok=True)
    input_dict = {"path": "/tmp/test_dir"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    os.rmdir("/tmp/test_dir")

    # Input 3: Non-existent path
    input_dict = {"path": "/tmp/non_existent_file.txt"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty string path
    input_dict = {"path": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Path with special characters
    with open("/tmp/file with spaces and !@#$.txt", "w") as f:
        f.write("Special characters test")
    input_dict = {"path": "/tmp/file with spaces and !@#$.txt"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    os.remove("/tmp/file with spaces and !@#$.txt")

    # Input 6: Path with unicode characters
    with open("/tmp/文件.txt", "w") as f:
        f.write("Unicode test")
    input_dict = {"path": "/tmp/文件.txt"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    os.remove("/tmp/文件.txt")

    # Input 7: Relative path
    with open("relative_file.txt", "w") as f:
        f.write("relative path test")
    input_dict = {"path": "relative_file.txt"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    os.remove("relative_file.txt")

    # Input 8: Path to a symbolic link (if supported by the OS)
    if os.name != 'nt':
        os.symlink("/tmp", "/tmp/symlink_to_tmp")
        input_dict = {"path": "/tmp/symlink_to_tmp"}
        list_of_inputs.append(copy.deepcopy(input_dict))
        os.remove("/tmp/symlink_to_tmp")

    # Input 9: Long path
    long_path = "/tmp/" + "a" * 200
    os.makedirs(long_path, exist_ok=True)
    input_dict = {"path": long_path}
    list_of_inputs.append(copy.deepcopy(input_dict))
    os.rmdir(long_path)

    # Input 10: Path with backslashes (Windows) - attempt to create it but use forward slashes in the input

    input_dict = {"path": "C:/Windows/System32/cmd.exe"}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.gfile.exists"] = tf_io_gfile_exists_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.gfile.exists' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.gfile.exists'.")

check_valid('tf.io.gfile.exists', generated_inputs['tf.io.gfile.exists'], lib="tf", suffix=0)
