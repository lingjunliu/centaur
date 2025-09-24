
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_io_gfile_mkdir_inputs():
    list_of_inputs = []

    # Input 1: Simple path
    path = "my_directory"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple path 2
    path = "another_directory"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Path with dot
    path = "./my_directory"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Path with numbers
    path = "my_directory123"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Path with mixed characters
    path = "MyDir123"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Long path
    path = "a" * 100
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Path with special characters
    path = "dir_!@#$"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Unicode path
    path = "你好世界"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: another simple path
    path = "test_dir"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: One more simple path
    path = "sample_dir"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.gfile.mkdir"] = tf_io_gfile_mkdir_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.gfile.mkdir' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.gfile.mkdir'.")

check_valid('tf.io.gfile.mkdir', generated_inputs['tf.io.gfile.mkdir'], lib="tf", suffix=0)
