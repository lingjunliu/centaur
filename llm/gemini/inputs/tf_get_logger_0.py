
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np
import logging

def tf_get_logger_inputs():
    list_of_inputs = []

    # There are no inputs required by the API. The goal is to showcase usage of the logger

    # Example 1: Basic usage
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Setting log level to ERROR
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Logging a warning
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Logging an error
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Logging an info message
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 6: Logging with string formatting
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 7: Logging a debug message (likely not shown by default)
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 8: Setting level to DEBUG
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 9: Logging a fatal message.
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 10: Setting level to INFO
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.get_logger"] = tf_get_logger_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.get_logger' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.get_logger'.")

check_valid('tf.get_logger', generated_inputs['tf.get_logger'], lib="tf", suffix=0)
