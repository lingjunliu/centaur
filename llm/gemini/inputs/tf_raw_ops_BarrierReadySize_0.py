
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_barrier_ready_size_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.constant("barrier_handle_1", dtype=tf.string)
    input_dict = {"handle": handle, "name": "barrier_ready_size_op_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.constant("barrier_handle_2", dtype=tf.string)
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.constant("another_barrier", dtype=tf.string)
    input_dict = {"handle": handle, "name": "specific_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.constant("yet_another_barrier", dtype=tf.string)
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.constant("barrier5", dtype=tf.string)
    input_dict = {"handle": handle, "name": "b5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.constant("barrier6", dtype=tf.string)
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = tf.constant("long_barrier_name_7", dtype=tf.string)
    input_dict = {"handle": handle, "name": "long_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.constant("barrier_8", dtype=tf.string)
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.constant("a9", dtype=tf.string)
    input_dict = {"handle": handle, "name": "a9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = tf.constant("b10", dtype=tf.string)
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    handle = tf.constant("barrier11", dtype=tf.string)
    input_dict = {"handle": handle, "name": "test"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.BarrierReadySize"] = tf_raw_ops_barrier_ready_size_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.BarrierReadySize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BarrierReadySize'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.BarrierReadySize', generated_inputs['tf.raw_ops.BarrierReadySize'], lib="tf", suffix=0)
