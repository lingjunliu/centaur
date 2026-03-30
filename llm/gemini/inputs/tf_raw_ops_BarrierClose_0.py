
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_barrierclose_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.Variable(np.array("handle1").astype(np.object_), dtype=tf.string)
    cancel_pending_enqueues = False
    name = "barrier_close_1"
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.Variable(np.array("handle2").astype(np.object_), dtype=tf.string)
    cancel_pending_enqueues = True
    name = "barrier_close_2"
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.Variable(np.array("very_long_handle_name_3").astype(np.object_), dtype=tf.string)
    cancel_pending_enqueues = False
    name = None
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.Variable(np.array("handle4").astype(np.object_), dtype=tf.string)
    cancel_pending_enqueues = True
    name = ""
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    handle = tf.Variable(np.array("handle5").astype(np.object_), dtype=tf.string)
    cancel_pending_enqueues = False
    name = "barrier_close_5"
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.Variable(np.array("handle6").astype(np.object_), dtype=tf.string)
    cancel_pending_enqueues = True
    name = "barrier_close_6"
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = tf.Variable(np.array("handle7").astype(np.object_), dtype=tf.string)
    cancel_pending_enqueues = False
    name = "barrier_close_7"
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.Variable(np.array("handle8").astype(np.object_), dtype=tf.string)
    cancel_pending_enqueues = True
    name = "barrier_close_8"
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.Variable(np.array("handle9").astype(np.object_), dtype=tf.string)
    cancel_pending_enqueues = False
    name = "barrier_close_9"
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = tf.Variable(np.array("handle10").astype(np.object_), dtype=tf.string)
    cancel_pending_enqueues = True
    name = "barrier_close_10"
    input_dict = {"handle": handle, "cancel_pending_enqueues": cancel_pending_enqueues, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.BarrierClose"] = tf_raw_ops_barrierclose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.BarrierClose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BarrierClose'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.BarrierClose', generated_inputs['tf.raw_ops.BarrierClose'], lib="tf", suffix=0)
