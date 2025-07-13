
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_raw_ops_ControlTrigger_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {"name": "control_trigger_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {"name": "another_trigger"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {"name": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {"name": "trigger_with_numbers_123"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {"name": "trigger_with_symbols_safe"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {"name": "very_long_trigger_name_" * 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {"name": "camelCaseTrigger"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {"name": "snake_case_trigger"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {"name": "mixedCase_123Trigger"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {"name": "triggerWithHyphen"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ControlTrigger"] = tf_raw_ops_ControlTrigger_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ControlTrigger' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ControlTrigger'.")

check_valid('tf.raw_ops.ControlTrigger', generated_inputs['tf.raw_ops.ControlTrigger'], lib="tf", suffix=0)
