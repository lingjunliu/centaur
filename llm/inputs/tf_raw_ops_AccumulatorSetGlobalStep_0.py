
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_accumulator_set_global_step_inputs():
    list_of_inputs = []

    # Input 1
    handle = np.array(b"accumulator_handle_1", dtype=np.dtype('string'))
    new_global_step = np.array(100, dtype=np.int64)
    input_dict = {"handle": handle, "new_global_step": new_global_step, "name": "set_global_step_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = np.array(b"accumulator_handle_2", dtype=np.dtype('string'))
    new_global_step = np.array(0, dtype=np.int64)
    input_dict = {"handle": handle, "new_global_step": new_global_step, "name": "set_global_step_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = np.array(b"accumulator_handle_3", dtype=np.dtype('string'))
    new_global_step = np.array(-1, dtype=np.int64)
    input_dict = {"handle": handle, "new_global_step": new_global_step, "name": "set_global_step_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = np.array(b"accumulator_handle_4", dtype=np.dtype('string'))
    new_global_step = np.array(2**31 - 1, dtype=np.int64)
    input_dict = {"handle": handle, "new_global_step": new_global_step, "name": "set_global_step_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    handle = np.array(b"accumulator_handle_5", dtype=np.dtype('string'))
    new_global_step = np.array(-(2**31), dtype=np.int64)
    input_dict = {"handle": handle, "new_global_step": new_global_step, "name": "set_global_step_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = np.array(b"accumulator_handle_6", dtype=np.dtype('string'))
    new_global_step = np.array(1, dtype=np.int64)
    input_dict = {"handle": handle, "new_global_step": new_global_step, "name": "set_global_step_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = np.array(b"long_accumulator_handle_7", dtype=np.dtype('string'))
    new_global_step = np.array(9999999999, dtype=np.int64)
    input_dict = {"handle": handle, "new_global_step": new_global_step, "name": "set_global_step_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = np.array(b"", dtype=np.dtype('string'))
    new_global_step = np.array(12345, dtype=np.int64)
    input_dict = {"handle": handle, "new_global_step": new_global_step, "name": "set_global_step_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = np.array(b"accumulator_handle_9", dtype=np.dtype('string'))
    new_global_step = np.array(-12345, dtype=np.int64)
    input_dict = {"handle": handle, "new_global_step": new_global_step, "name": "set_global_step_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = np.array(b"12345", dtype=np.dtype('string'))
    new_global_step = np.array(67890, dtype=np.int64)
    input_dict = {"handle": handle, "new_global_step": new_global_step, "name": "set_global_step_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AccumulatorSetGlobalStep"] = tf_raw_ops_accumulator_set_global_step_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.AccumulatorSetGlobalStep' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AccumulatorSetGlobalStep'.")

check_valid('tf.raw_ops.AccumulatorSetGlobalStep', generated_inputs['tf.raw_ops.AccumulatorSetGlobalStep'], lib="tf", suffix=0)
