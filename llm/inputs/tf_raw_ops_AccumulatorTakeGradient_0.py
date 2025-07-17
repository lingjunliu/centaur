
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_AccumulatorTakeGradient_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.constant("handle1", dtype=tf.string)
    num_required = tf.constant(2, dtype=tf.int32)
    dtype = tf.float32
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.constant("handle2", dtype=tf.string)
    num_required = tf.constant(5, dtype=tf.int32)
    dtype = tf.float64
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.constant("handle3", dtype=tf.string)
    num_required = tf.constant(10, dtype=tf.int32)
    dtype = tf.int32
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.constant("handle4", dtype=tf.string)
    num_required = tf.constant(1, dtype=tf.int32)
    dtype = tf.uint8
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.constant("handle5", dtype=tf.string)
    num_required = tf.constant(3, dtype=tf.int32)
    dtype = tf.int16
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.constant("handle6", dtype=tf.string)
    num_required = tf.constant(7, dtype=tf.int32)
    dtype = tf.int8
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = tf.constant("handle7", dtype=tf.string)
    num_required = tf.constant(4, dtype=tf.int32)
    dtype = tf.complex64
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.constant("handle8", dtype=tf.string)
    num_required = tf.constant(6, dtype=tf.int32)
    dtype = tf.int64
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.constant("handle9", dtype=tf.string)
    num_required = tf.constant(8, dtype=tf.int32)
    dtype = tf.bfloat16
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = tf.constant("handle10", dtype=tf.string)
    num_required = tf.constant(9, dtype=tf.int32)
    dtype = tf.half
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AccumulatorTakeGradient"] = tf_raw_ops_AccumulatorTakeGradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.AccumulatorTakeGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AccumulatorTakeGradient'.")

check_valid('tf.raw_ops.AccumulatorTakeGradient', generated_inputs['tf.raw_ops.AccumulatorTakeGradient'], lib="tf", suffix=0)
