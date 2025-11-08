
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import torch
import copy

def tf_raw_ops_abort_inputs():
    list_of_inputs = []

    # Input 1
    error_msg = ""
    exit_without_error = False
    name = "abort_op_empty_msg"
    input_dict = {
        "error_msg": error_msg,
        "exit_without_error": exit_without_error,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    error_msg = "Abort now"
    exit_without_error = True
    name = "abort_ok_exit"
    input_dict = {
        "error_msg": error_msg,
        "exit_without_error": exit_without_error,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    error_msg = "Early termination requested"
    exit_without_error = False
    name = "early_termination"
    input_dict = {
        "error_msg": error_msg,
        "exit_without_error": exit_without_error,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    error_msg = "Line1\nLine2\tTabbed"
    exit_without_error = False
    name = "with_newlines"
    input_dict = {
        "error_msg": error_msg,
        "exit_without_error": exit_without_error,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    error_msg = "x" * 1024
    exit_without_error = True
    name = "very_long_error_msg"
    input_dict = {
        "error_msg": error_msg,
        "exit_without_error": exit_without_error,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    error_msg = "Ошибка завершения процесса"
    exit_without_error = False
    name = "cyrillic_name"
    input_dict = {
        "error_msg": error_msg,
        "exit_without_error": exit_without_error,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    error_msg = "終了します"
    exit_without_error = True
    name = "japanese_message"
    input_dict = {
        "error_msg": error_msg,
        "exit_without_error": exit_without_error,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    error_msg = "Aborting due to invalid state: -1"
    exit_without_error = False
    name = "negative_state"
    input_dict = {
        "error_msg": error_msg,
        "exit_without_error": exit_without_error,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    error_msg = "Special chars !@#$%^&*()[]{};:,.<>/?|`~"
    exit_without_error = True
    name = "special_chars"
    input_dict = {
        "error_msg": error_msg,
        "exit_without_error": exit_without_error,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    error_msg = "CaseSensitiveMessage"
    exit_without_error = False
    name = "MixedCaseName"
    input_dict = {
        "error_msg": error_msg,
        "exit_without_error": exit_without_error,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Abort"] = tf_raw_ops_abort_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Abort' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Abort'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Abort', generated_inputs['tf.raw_ops.Abort'], lib="tf", suffix=0)
