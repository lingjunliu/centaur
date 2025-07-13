
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_BarrierClose_inputs():
    list_of_inputs = []

    # Input 1
    handle = "barrier_handle"
    cancel_pending_enqueues = np.bool_(False)
    name = "close_barrier_1"
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = "another_barrier"
    cancel_pending_enqueues = np.bool_(True)
    name = None
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = "yet_another_barrier"
    cancel_pending_enqueues = np.bool_(False)
    name = ""
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = "test_barrier_123"
    cancel_pending_enqueues = np.bool_(True)
    name = "close_barrier_4"
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = "barrier_456"
    cancel_pending_enqueues = np.bool_(False)
    name = "close_barrier_5"
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = "long_barrier_name_789"
    cancel_pending_enqueues = np.bool_(True)
    name = "close_barrier_6"
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = "barrier_alpha"
    cancel_pending_enqueues = np.bool_(False)
    name = "close_barrier_7"
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = "barrier_beta"
    cancel_pending_enqueues = np.bool_(True)
    name = "close_barrier_8"
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = "barrier_gamma"
    cancel_pending_enqueues = np.bool_(False)
    name = "close_barrier_9"
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = "barrier_delta"
    cancel_pending_enqueues = np.bool_(True)
    name = "close_barrier_10"
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.BarrierClose"] = tf_raw_ops_BarrierClose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.BarrierClose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BarrierClose'.")

check_valid('tf.raw_ops.BarrierClose', generated_inputs['tf.raw_ops.BarrierClose'], lib="tf", suffix=0)
