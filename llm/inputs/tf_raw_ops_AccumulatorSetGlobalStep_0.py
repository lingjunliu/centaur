
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_accumulator_set_global_step_inputs():
    list_of_inputs = []

    # Input 1
    accumulator = tf.compat.v1.accumulator(dtype=tf.float32, shape=[])
    handle = accumulator.resource_handle
    new_global_step = tf.constant(10, dtype=tf.int64)
    name = "set_global_step_op_1"

    input_dict = {
        "handle": handle,
        "new_global_step": new_global_step,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    accumulator = tf.compat.v1.accumulator(dtype=tf.float32, shape=[])
    handle = accumulator.resource_handle
    new_global_step = tf.constant(0, dtype=tf.int64)
    name = None

    input_dict = {
        "handle": handle,
        "new_global_step": new_global_step,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    accumulator = tf.compat.v1.accumulator(dtype=tf.float32, shape=[])
    handle = accumulator.resource_handle
    new_global_step = tf.constant(-5, dtype=tf.int64)
    name = "negative_step"

    input_dict = {
        "handle": handle,
        "new_global_step": new_global_step,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    accumulator = tf.compat.v1.accumulator(dtype=tf.float32, shape=[])
    handle = accumulator.resource_handle
    new_global_step = tf.constant(2**31 - 1, dtype=tf.int64)  # Max int32
    name = "max_int32_step"

    input_dict = {
        "handle": handle,
        "new_global_step": new_global_step,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    accumulator = tf.compat.v1.accumulator(dtype=tf.float32, shape=[])
    handle = accumulator.resource_handle
    new_global_step = tf.constant(2**63 - 1, dtype=tf.int64)  # Max int64
    name = "max_int64_step"

    input_dict = {
        "handle": handle,
        "new_global_step": new_global_step,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    accumulator = tf.compat.v1.accumulator(dtype=tf.float32, shape=[])
    handle = accumulator.resource_handle
    new_global_step = tf.constant(np.array(15, dtype=np.int64))
    name = "numpy_step"

    input_dict = {
        "handle": handle,
        "new_global_step": new_global_step,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    accumulator = tf.compat.v1.accumulator(dtype=tf.float32, shape=[])
    handle = accumulator.resource_handle
    new_global_step = tf.constant(-2**63, dtype=tf.int64)  # Min int64
    name = "min_int64_step"

    input_dict = {
        "handle": handle,
        "new_global_step": new_global_step,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    accumulator = tf.compat.v1.accumulator(dtype=tf.float32, shape=[])
    handle = accumulator.resource_handle
    new_global_step = tf.constant(1, dtype=tf.int64)
    name = ""

    input_dict = {
        "handle": handle,
        "new_global_step": new_global_step,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    accumulator = tf.compat.v1.accumulator(dtype=tf.float32, shape=[])
    handle = accumulator.resource_handle
    new_global_step = tf.constant(123456789, dtype=tf.int64)
    name = "some_long_name"

    input_dict = {
        "handle": handle,
        "new_global_step": new_global_step,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    accumulator = tf.compat.v1.accumulator(dtype=tf.float32, shape=[])
    handle = accumulator.resource_handle
    new_global_step = tf.constant(-123456789, dtype=tf.int64)
    name = "some_long_negative_name"

    input_dict = {
        "handle": handle,
        "new_global_step": new_global_step,
        "name": name
    }
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
