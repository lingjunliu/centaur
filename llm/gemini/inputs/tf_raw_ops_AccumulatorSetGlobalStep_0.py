
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_accumulator_set_global_step_inputs():
    list_of_inputs = []

    # Input 1
    handle_str = "accumulator_handle_1"
    handle = tf.constant(handle_str, dtype=tf.string)
    new_global_step = tf.constant(10, dtype=tf.int64)
    input_dict = {"handle": handle, "new_global_step": new_global_step, "name": "op_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle_str = "accumulator_handle_2"
    handle = tf.constant(handle_str, dtype=tf.string)
    new_global_step = tf.constant(100, dtype=tf.int64)
    input_dict = {"handle": handle, "new_global_step": new_global_step, "name": "op_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle_str = "accumulator_handle_3"
    handle = tf.constant(handle_str, dtype=tf.string)
    new_global_step = tf.constant(0, dtype=tf.int64)
    input_dict = {"handle": handle, "new_global_step": new_global_step, "name": "op_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle_str = "accumulator_handle_4"
    handle = tf.constant(handle_str, dtype=tf.string)
    new_global_step = tf.constant(-10, dtype=tf.int64)
    input_dict = {"handle": handle, "new_global_step": new_global_step, "name": "op_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle_str = "accumulator_handle_5"
    handle = tf.constant(handle_str, dtype=tf.string)
    new_global_step = tf.constant(2**31 - 1, dtype=tf.int64)
    input_dict = {"handle": handle, "new_global_step": new_global_step, "name": "op_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle_str = "accumulator_handle_6"
    handle = tf.constant(handle_str, dtype=tf.string)
    new_global_step = tf.constant(-(2**31), dtype=tf.int64)
    input_dict = {"handle": handle, "new_global_step": new_global_step, "name": "op_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle_str = "accumulator_handle_7"
    handle = tf.constant(handle_str, dtype=tf.string)
    new_global_step = tf.constant(5, dtype=tf.int64)
    input_dict = {"handle": handle, "new_global_step": new_global_step, "name": "op_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle_str = "accumulator_handle_8"
    handle = tf.constant(handle_str, dtype=tf.string)
    new_global_step = tf.constant(1000, dtype=tf.int64)
    input_dict = {"handle": handle, "new_global_step": new_global_step, "name": "op_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    handle_str = "accumulator_handle_9"
    handle = tf.constant(handle_str, dtype=tf.string)
    new_global_step = tf.constant(-1000, dtype=tf.int64)
    input_dict = {"handle": handle, "new_global_step": new_global_step, "name": "op_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    handle_str = "accumulator_handle_10"
    handle = tf.constant(handle_str, dtype=tf.string)
    new_global_step = tf.constant(0, dtype=tf.int64)
    input_dict = {"handle": handle, "new_global_step": new_global_step, "name": "op_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AccumulatorSetGlobalStep"] = tf_raw_ops_accumulator_set_global_step_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.AccumulatorSetGlobalStep' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AccumulatorSetGlobalStep'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.AccumulatorSetGlobalStep', generated_inputs['tf.raw_ops.AccumulatorSetGlobalStep'], lib="tf", suffix=0)
