
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_compat_path_to_str_inputs():
    list_of_inputs = []

    # Input 1: Simple path
    path = "path/to/file.txt"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Path with special characters
    path = "./../my_folder/file_with_!@#$%^&*()_+=-`~.txt"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Absolute path
    path = "/absolute/path/to/the/file.dat"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Windows path
    path = "C:\\Users\\user\\Documents\\file.csv"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty string
    path = ""
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Path with Unicode characters
    path = "路径/到/文件.txt"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Path with multiple consecutive slashes
    path = "path///to//file.txt"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: A path with a trailing slash
    path = "path/to/directory/"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: A very long path
    path = "a" * 2000
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Path with spaces
    path = "path with spaces/file.txt"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.compat.path_to_str"] = tf_compat_path_to_str_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.compat.path_to_str' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.compat.path_to_str'.")

check_valid('tf.compat.path_to_str', generated_inputs['tf.compat.path_to_str'], lib="tf", suffix=0)
