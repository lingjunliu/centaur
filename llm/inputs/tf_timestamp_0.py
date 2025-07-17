
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_timestamp_inputs():
    list_of_inputs = []

    # Input 1: No name (None)
    input_dict = {"name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple name
    input_dict = {"name": "timestamp_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Name with underscores
    input_dict = {"name": "time_stamp_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Another simple name
    input_dict = {"name": "my_time"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty string
    input_dict = {"name": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.timestamp"] = tf_timestamp_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.timestamp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.timestamp'.")

check_valid('tf.timestamp', generated_inputs['tf.timestamp'], lib="tf", suffix=0)
