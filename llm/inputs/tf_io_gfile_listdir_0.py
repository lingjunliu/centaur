
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np
import os

def tf_io_gfile_listdir_inputs():
    list_of_inputs = []

    # Input 1: Basic valid path
    input_dict = {"path": "."}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Empty string path (might be valid in some contexts)
    input_dict = {"path": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Absolute path (if applicable, depends on environment)
    input_dict = {"path": os.getcwd()}
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
