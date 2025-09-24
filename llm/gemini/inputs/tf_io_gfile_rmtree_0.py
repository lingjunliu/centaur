
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np
import os

def tf_io_gfile_rmtree_inputs():
    list_of_inputs = []

    # Helper function to create directories
    def create_dir(path):
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as error:
            print(f"Error creating directory {path}: {error}")

    # Input 1: Simple path
    path1 = "./test_dir"
    create_dir(path1)
    input_dict = {"path": path1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Nested path
    path2 = "./nested/directory"
    create_dir(path2)
    input_dict = {"path": "./nested/directory"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Path with numbers
    path3 = "./directory123"
    create_dir(path3)
    input_dict = {"path": "./directory123"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Trailing slash
    path4 = "./directory/"
    create_dir(path4)
    input_dict = {"path": "./directory/"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Unicode characters
    path5 = "./目录"
    create_dir(path5)
    input_dict = {"path": "./目录"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Short name
    path6 = "./a"
    create_dir(path6)
    input_dict = {"path": "./a"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex name
    path7 = "./complex_dir_name"
    create_dir(path7)
    input_dict = {"path": "./complex_dir_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Spaces in path
    path8 = "./path with spaces"
    create_dir(path8)
    input_dict = {"path": "./path with spaces"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: dot slash
    path9 = "./another_dir"
    create_dir(path9)
    input_dict = {"path": "./another_dir"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Many subdirectories
    path10 = "./many/sub/directories"
    create_dir(path10)
    input_dict = {"path": path10}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.gfile.rmtree"] = tf_io_gfile_rmtree_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.gfile.rmtree' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.gfile.rmtree'.")

check_valid('tf.io.gfile.rmtree', generated_inputs['tf.io.gfile.rmtree'], lib="tf", suffix=0)
