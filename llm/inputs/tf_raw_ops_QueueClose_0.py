
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QueueClose_inputs():
    list_of_inputs = []

    # Input 1
    handle = np.array("queue_handle").astype(np.string_)
    cancel_pending_enqueues = False
    name = "close_queue_1"
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = np.array("another_queue").astype(np.string_)
    cancel_pending_enqueues = True
    name = None
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = np.array("yet_another_queue").astype(np.string_)
    cancel_pending_enqueues = False
    name = "close_queue_3"
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = np.array("a_very_long_queue_name").astype(np.string_)
    cancel_pending_enqueues = True
    name = "long_name_queue"
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = np.array("short").astype(np.string_)
    cancel_pending_enqueues = False
    name = None
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = np.array("").astype(np.string_)
    cancel_pending_enqueues = True
    name = "empty_queue"
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = np.array("queue_7").astype(np.string_)
    cancel_pending_enqueues = True
    name = None
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = np.array("queue_8").astype(np.string_)
    cancel_pending_enqueues = False
    name = "a_name"
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    handle = np.array("queue_9").astype(np.string_)
    cancel_pending_enqueues = True
    name = "another_name"
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    handle = np.array("queue_10").astype(np.string_)
    cancel_pending_enqueues = False
    name = None
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QueueClose"] = tf_raw_ops_QueueClose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QueueClose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QueueClose'.")

check_valid('tf.raw_ops.QueueClose', generated_inputs['tf.raw_ops.QueueClose'], lib="tf", suffix=0)
