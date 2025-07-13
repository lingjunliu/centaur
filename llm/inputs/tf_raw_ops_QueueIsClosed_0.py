
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_queueisclosed_inputs():
    list_of_inputs = []

    # Input 1
    handle = np.array("queue_handle_1", dtype=np.string_)
    name = None
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = np.array("queue_handle_2", dtype=np.string_)
    name = "QueueIsClosedOp_2"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = np.array("queue_handle_3", dtype=np.string_)
    name = ""
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    handle = np.array("queue_handle_4_long_name", dtype=np.string_)
    name = "ThisIsALongNameForAnOp"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = np.array("queue_handle_5", dtype=np.string_)
    name = "5"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = np.array("queue_handle_6", dtype=np.string_)
    name = "queue_is_closed_6"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    handle = np.array("queue_handle_7", dtype=np.string_)
    name = "7_with_underscore"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = np.array("queue_handle_8", dtype=np.string_)
    name = "8WithCamelCase"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = np.array("queue_handle_9", dtype=np.string_)
    name = "9.With.Dots"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    handle = np.array("queue_handle_10", dtype=np.string_)
    name = "10-With-Hyphens"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QueueIsClosed"] = tf_raw_ops_queueisclosed_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QueueIsClosed' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QueueIsClosed'.")

check_valid('tf.raw_ops.QueueIsClosed', generated_inputs['tf.raw_ops.QueueIsClosed'], lib="tf", suffix=0)
