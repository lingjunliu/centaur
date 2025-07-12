
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_raw_ops_abort_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "error_msg": "",
        "exit_without_error": False,
        "name": "abort_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "error_msg": "This is an error message",
        "exit_without_error": False,
        "name": "abort_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "error_msg": "Another error message",
        "exit_without_error": True,
        "name": "abort_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "error_msg": "",
        "exit_without_error": True,
        "name": "abort_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "error_msg": "A very long error message with many characters.",
        "exit_without_error": False,
        "name": "abort_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "error_msg": "Special characters: !@#$%^&*()_+",
        "exit_without_error": True,
        "name": "abort_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "error_msg": "",
        "exit_without_error": False,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "error_msg": "Unicode characters: こんにちは",
        "exit_without_error": False,
        "name": "abort_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "error_msg": "Error with numbers: 1234567890",
        "exit_without_error": True,
        "name": "abort_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_dict = {
        "error_msg": "Empty String",
        "exit_without_error": False,
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    input_dict = {
        "error_msg": "A very specific error",
        "exit_without_error": True,
        "name": "A_Specific_Abort"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Abort"] = tf_raw_ops_abort_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Abort' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Abort'.")

check_valid('tf.raw_ops.Abort', generated_inputs['tf.raw_ops.Abort'], lib="tf", suffix=0)
