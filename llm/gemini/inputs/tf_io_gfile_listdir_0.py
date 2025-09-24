
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np
import os

def tf_io_gfile_listdir_inputs():
    list_of_inputs = []

    # Input 1: Valid path to an existing directory
    dir_name = "temp_dir_1"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if not os.path.exists(os.path.join(dir_name, "file1.txt")):
        with open(os.path.join(dir_name, "file1.txt"), "w") as f:
            f.write("test")
    path = dir_name
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Path to an empty directory
    dir_name = "empty_dir_2"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    path = dir_name
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Path with a trailing slash
    dir_name = "trailing_slash_dir_3"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if not os.path.exists(os.path.join(dir_name, "file1.txt")):
        with open(os.path.join(dir_name, "file1.txt"), "w") as f:
            f.write("test")

    path = dir_name + "/"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Path with multiple files
    dir_name = "multiple_files_dir_4"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if not os.path.exists(os.path.join(dir_name, "file1.txt")):
        with open(os.path.join(dir_name, "file1.txt"), "w") as f:
            f.write("test")
    if not os.path.exists(os.path.join(dir_name, "file2.txt")):
        with open(os.path.join(dir_name, "file2.txt"), "w") as f:
            f.write("test")

    path = dir_name
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Path with spaces in name
    dir_name = "dir with spaces_5"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if not os.path.exists(os.path.join(dir_name, "file1.txt")):
        with open(os.path.join(dir_name, "file1.txt"), "w") as f:
            f.write("test")
    path = dir_name
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Path with special characters
    dir_name = "special_chars_dir_6"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if not os.path.exists(os.path.join(dir_name, "!@#$%^&*.txt")):
        with open(os.path.join(dir_name, "!@#$%^&*.txt"), "w") as f:
            f.write("test")
    path = dir_name
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Path as raw string (using r prefix)
    dir_name = "raw_string_dir_7"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if not os.path.exists(os.path.join(dir_name, "file.txt")):
        with open(os.path.join(dir_name, "file.txt"), "w") as f:
            f.write("test")
    path = dir_name
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Relative path
    dir_name = "relative_dir_8"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if not os.path.exists(os.path.join(dir_name, "file.txt")):
        with open(os.path.join(dir_name, "file.txt"), "w") as f:
            f.write("test")
    path = "./" + dir_name
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    dir_name = "unicode_dir_9"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if not os.path.exists(os.path.join(dir_name, "文件.txt")):
        with open(os.path.join(dir_name, "文件.txt"), "w") as f:
            f.write("test")
    path = dir_name
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    dir_name = "long_name_dir_10"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    long_file_name = "file_" + "a" * 200 + ".txt"
    if not os.path.exists(os.path.join(dir_name, long_file_name)):
        with open(os.path.join(dir_name, long_file_name), "w") as f:
            f.write("test")

    path = dir_name
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.gfile.listdir"] = tf_io_gfile_listdir_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.gfile.listdir' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.gfile.listdir'.")

check_valid('tf.io.gfile.listdir', generated_inputs['tf.io.gfile.listdir'], lib="tf", suffix=0)
