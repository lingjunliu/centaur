
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_queueclose_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.constant("queue_handle_1", dtype=tf.string).numpy()
    cancel_pending_enqueues = False
    name = "queue_close_op_1"
    input_dict = {"handle": handle, "cancel_pending_enqueues": np.bool_(cancel_pending_enqueues), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.constant("queue_handle_2", dtype=tf.string).numpy()
    cancel_pending_enqueues = True
    name = "queue_close_op_2"
    input_dict = {"handle": handle, "cancel_pending_enqueues": np.bool_(cancel_pending_enqueues), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.constant("queue_handle_3", dtype=tf.string).numpy()
    cancel_pending_enqueues = False
    name = None
    input_dict = {"handle": handle, "cancel_pending_enqueues": np.bool_(cancel_pending_enqueues), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.constant("queue_handle_4", dtype=tf.string).numpy()
    cancel_pending_enqueues = True
    name = ""
    input_dict = {"handle": handle, "cancel_pending_enqueues": np.bool_(cancel_pending_enqueues), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.constant("queue_handle_5", dtype=tf.string).numpy()
    cancel_pending_enqueues = False
    name = "queue_close_op_5"
    input_dict = {"handle": handle, "cancel_pending_enqueues": np.bool_(cancel_pending_enqueues), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.constant("queue_handle_6", dtype=tf.string).numpy()
    cancel_pending_enqueues = True
    name = "queue_close_op_6"
    input_dict = {"handle": handle, "cancel_pending_enqueues": np.bool_(cancel_pending_enqueues), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = tf.constant("queue_handle_7", dtype=tf.string).numpy()
    cancel_pending_enqueues = False
    name = "very_long_queue_name_7"
    input_dict = {"handle": handle, "cancel_pending_enqueues": np.bool_(cancel_pending_enqueues), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.constant("queue_handle_8", dtype=tf.string).numpy()
    cancel_pending_enqueues = True
    name = "queue_8"
    input_dict = {"handle": handle, "cancel_pending_enqueues": np.bool_(cancel_pending_enqueues), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.constant("handle_9", dtype=tf.string).numpy()
    cancel_pending_enqueues = False
    name = "op_name_9"
    input_dict = {"handle": handle, "cancel_pending_enqueues": np.bool_(cancel_pending_enqueues), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = tf.constant("handle_10", dtype=tf.string).numpy()
    cancel_pending_enqueues = True
    name = "operation_10"
    input_dict = {"handle": handle, "cancel_pending_enqueues": np.bool_(cancel_pending_enqueues), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QueueClose"] = tf_raw_ops_queueclose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QueueClose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QueueClose'.")

check_valid('tf.raw_ops.QueueClose', generated_inputs['tf.raw_ops.QueueClose'], lib="tf", suffix=0)
