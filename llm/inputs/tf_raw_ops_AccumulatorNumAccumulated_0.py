
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_accumulator_num_accumulated_inputs():
    list_of_inputs = []

    # Input 1
    handle = "accumulator_handle_1"
    input_dict = {"handle": handle, "name": "name_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = "accumulator_handle_2"
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = "accumulator_handle_3"
    input_dict = {"handle": handle, "name": "another_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = ""
    input_dict = {"handle": handle, "name": "empty_handle"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = " "
    input_dict = {"handle": handle, "name": "space_handle"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = "accumulator_handle_6"
    input_dict = {"handle": handle, "name": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = "accumulator_handle_7"
    input_dict = {"handle": handle, "name": "name_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = "accumulator_handle_8"
    input_dict = {"handle": handle, "name": "name_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = "accumulator_handle_9"
    input_dict = {"handle": handle, "name": "name_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = "accumulator_handle_10"
    input_dict = {"handle": handle, "name": "name_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AccumulatorNumAccumulated"] = tf_raw_ops_accumulator_num_accumulated_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.AccumulatorNumAccumulated' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AccumulatorNumAccumulated'.")

check_valid('tf.raw_ops.AccumulatorNumAccumulated', generated_inputs['tf.raw_ops.AccumulatorNumAccumulated'], lib="tf", suffix=0)
