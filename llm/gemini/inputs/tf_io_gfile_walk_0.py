
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np
import os

def tf_io_gfile_walk_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    top = "."
    topdown = True
    onerror = []
    input_dict = {"top": top, "topdown": topdown, "onerror": onerror}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: topdown = False
    top = "."
    topdown = False
    onerror = []
    input_dict = {"top": top, "topdown": topdown, "onerror": onerror}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: onerror handler
    def error_handler(error):
        print(f"Error: {error}")
    top = "."
    topdown = True
    onerror = [error_handler]
    input_dict = {"top": top, "topdown": topdown, "onerror": onerror}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: top as relative path
    top = "./"
    topdown = True
    onerror = []
    input_dict = {"top": top, "topdown": topdown, "onerror": onerror}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty directory
    if not os.path.exists("empty_dir"):
        os.makedirs("empty_dir")
    top = "empty_dir"
    topdown = True
    onerror = []
    input_dict = {"top": top, "topdown": topdown, "onerror": onerror}
    list_of_inputs.append(copy.deepcopy(input_dict))
    #Clean up the empty dir
    os.rmdir("empty_dir")

    # Input 6: Different onerror
    def another_error_handler(error):
        pass
    top = "."
    topdown = False
    onerror = [another_error_handler]
    input_dict = {"top": top, "topdown": topdown, "onerror": onerror}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Longer path
    top = "./."
    topdown = True
    onerror = []
    input_dict = {"top": top, "topdown": topdown, "onerror": onerror}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: Create temp dir
    if not os.path.exists("temp_dir"):
        os.makedirs("temp_dir")
    top = "temp_dir"
    topdown = False
    onerror = []
    input_dict = {"top": top, "topdown": topdown, "onerror": onerror}
    list_of_inputs.append(copy.deepcopy(input_dict))
    #Clean up
    os.rmdir("temp_dir")

    # Input 9: Path with special character
    top = "./"
    topdown = False
    onerror = []
    input_dict = {"top": top, "topdown": topdown, "onerror": onerror}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: onerror with empty list
    top = "."
    topdown = True
    onerror = []
    input_dict = {"top": top, "topdown": topdown, "onerror": onerror}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.gfile.walk"] = tf_io_gfile_walk_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.gfile.walk' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.gfile.walk'.")

check_valid('tf.io.gfile.walk', generated_inputs['tf.io.gfile.walk'], lib="tf", suffix=0)
