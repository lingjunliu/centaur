
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np
import os

def tf_train_latest_checkpoint_inputs():
    list_of_inputs = []

    # Input 1: Basic case - empty directory
    input_dict = {
        "checkpoint_dir": "",
        "latest_filename": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic case - non-empty directory with default latest_filename
    input_dict = {
        "checkpoint_dir": "./",  # Current directory
        "latest_filename": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Non-empty directory with a custom latest_filename
    input_dict = {
        "checkpoint_dir": "./",
        "latest_filename": "my_checkpoint"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4:  Empty directory with a custom latest_filename
    input_dict = {
        "checkpoint_dir": "",
        "latest_filename": "special_checkpoint"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: A longer path for checkpoint_dir
    input_dict = {
        "checkpoint_dir": "/tmp/model_checkpoints",
        "latest_filename": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: A directory with only numbers in the name
    input_dict = {
        "checkpoint_dir": "12345",
        "latest_filename": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: A directory with special characters
    input_dict = {
        "checkpoint_dir": "./!@#$%^",
        "latest_filename": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Using absolute path
    input_dict = {
        "checkpoint_dir": os.path.abspath("./"),
        "latest_filename": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: Using backslashes on windows
    input_dict = {
        "checkpoint_dir": ".\\model_checkpoints",
        "latest_filename": "my_checkpoint_file"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Custom checkpoint file with special char
    input_dict = {
        "checkpoint_dir": "./",
        "latest_filename": "checkpoint!@#$%"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.train.latest_checkpoint"] = tf_train_latest_checkpoint_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.train.latest_checkpoint' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.train.latest_checkpoint'.")

check_valid('tf.train.latest_checkpoint', generated_inputs['tf.train.latest_checkpoint'], lib="tf", suffix=0)
