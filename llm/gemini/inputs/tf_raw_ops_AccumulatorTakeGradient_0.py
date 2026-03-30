
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_AccumulatorTakeGradient_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.constant("test_handle_1", dtype=tf.string)
    num_required = tf.constant(1, dtype=tf.int32)
    dtype = tf.float32
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.constant("test_handle_2", dtype=tf.string)
    num_required = tf.constant(5, dtype=tf.int32)
    dtype = tf.float64
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.constant("test_handle_3", dtype=tf.string)
    num_required = tf.constant(10, dtype=tf.int32)
    dtype = tf.int32
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.constant("test_handle_4", dtype=tf.string)
    num_required = tf.constant(2, dtype=tf.int32)
    dtype = tf.complex64
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.constant("test_handle_5", dtype=tf.string)
    num_required = tf.constant(3, dtype=tf.int32)
    dtype = tf.int64
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.constant("test_handle_6", dtype=tf.string)
    num_required = tf.constant(4, dtype=tf.int32)
    dtype = tf.bfloat16
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    handle = tf.constant("test_handle_7", dtype=tf.string)
    num_required = tf.constant(15, dtype=tf.int32)
    dtype = tf.half
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.constant("test_handle_8", dtype=tf.string)
    num_required = tf.constant(7, dtype=tf.int32)
    dtype = tf.uint8
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.constant("test_handle_9", dtype=tf.string)
    num_required = tf.constant(8, dtype=tf.int32)
    dtype = tf.int8
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = tf.constant("test_handle_10", dtype=tf.string)
    num_required = tf.constant(9, dtype=tf.int32)
    dtype = tf.complex128
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AccumulatorTakeGradient"] = tf_raw_ops_AccumulatorTakeGradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.AccumulatorTakeGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AccumulatorTakeGradient'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.AccumulatorTakeGradient', generated_inputs['tf.raw_ops.AccumulatorTakeGradient'], lib="tf", suffix=0)
