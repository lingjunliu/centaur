
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_barrier_close_inputs():
    list_of_inputs = []

    # Input 1
    handle = np.array("barrier_handle").astype(np.object_)
    cancel_pending_enqueues = np.array(False).astype(np.bool_)
    name = None

    input_dict = {
        "cancel_pending_enqueues": cancel_pending_enqueues,
        "name": name,
        "handle": handle
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = np.array("another_barrier").astype(np.object_)
    cancel_pending_enqueues = np.array(True).astype(np.bool_)
    name = "close_barrier_op"

    input_dict = {
        "cancel_pending_enqueues": cancel_pending_enqueues,
        "name": name,
        "handle": handle
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = np.array("barrier3").astype(np.object_)
    cancel_pending_enqueues = np.array(False).astype(np.bool_)
    name = "close_op_3"

    input_dict = {
        "cancel_pending_enqueues": cancel_pending_enqueues,
        "name": name,
        "handle": handle
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = np.array("barrier4").astype(np.object_)
    cancel_pending_enqueues = np.array(True).astype(np.bool_)
    name = None

    input_dict = {
        "cancel_pending_enqueues": cancel_pending_enqueues,
        "name": name,
        "handle": handle
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = np.array("barrier5").astype(np.object_)
    cancel_pending_enqueues = np.array(False).astype(np.bool_)
    name = None

    input_dict = {
        "cancel_pending_enqueues": cancel_pending_enqueues,
        "name": name,
        "handle": handle
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = np.array("barrier6").astype(np.object_)
    cancel_pending_enqueues = np.array(True).astype(np.bool_)
    name = "closing_time"

    input_dict = {
        "cancel_pending_enqueues": cancel_pending_enqueues,
        "name": name,
        "handle": handle
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = np.array("barrier7").astype(np.object_)
    cancel_pending_enqueues = np.array(False).astype(np.bool_)
    name = "barrier_7_close"

    input_dict = {
        "cancel_pending_enqueues": cancel_pending_enqueues,
        "name": name,
        "handle": handle
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = np.array("barrier8").astype(np.object_)
    cancel_pending_enqueues = np.array(True).astype(np.bool_)
    name = "close8"

    input_dict = {
        "cancel_pending_enqueues": cancel_pending_enqueues,
        "name": name,
        "handle": handle
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = np.array("barrier9").astype(np.object_)
    cancel_pending_enqueues = np.array(False).astype(np.bool_)
    name = None

    input_dict = {
        "cancel_pending_enqueues": cancel_pending_enqueues,
        "name": name,
        "handle": handle
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = np.array("barrier10").astype(np.object_)
    cancel_pending_enqueues = np.array(True).astype(np.bool_)
    name = None

    input_dict = {
        "cancel_pending_enqueues": cancel_pending_enqueues,
        "name": name,
        "handle": handle
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.BarrierClose"] = tf_raw_ops_barrier_close_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.BarrierClose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BarrierClose'.")

check_valid('tf.raw_ops.BarrierClose', generated_inputs['tf.raw_ops.BarrierClose'], lib="tf", suffix=0)
