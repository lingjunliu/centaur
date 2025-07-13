
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_queue_size_inputs():
    list_of_inputs = []

    # Input 1
    handle = np.array("queue_handle_1", dtype=np.object_)
    name = "queue_size_op_1"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = np.array("queue_handle_2", dtype=np.object_)
    name = None
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = np.array("another_queue", dtype=np.object_)
    name = "size_check"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = np.array("empty_queue", dtype=np.object_)
    name = ""
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    handle = np.array("queue_with_long_name", dtype=np.object_)
    name = "queue_size_with_long_name"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = np.array("queue_number_6", dtype=np.object_)
    name = "size_number_6"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = np.array("queue_name_7", dtype=np.object_)
    name = "queue_7_size"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = np.array("queue_handle_8", dtype=np.object_)
    name = "op_name_8"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = np.array("queue_handle_9", dtype=np.object_)
    name = "size_op_9"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = np.array("queue_handle_10", dtype=np.object_)
    name = "op_10"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    handle = np.array("queue_handle_11", dtype=np.object_)
    name = "very_long_operation_name_11"
    input_dict = {"handle": handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QueueSize"] = tf_raw_ops_queue_size_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QueueSize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QueueSize'.")

check_valid('tf.raw_ops.QueueSize', generated_inputs['tf.raw_ops.QueueSize'], lib="tf", suffix=0)
