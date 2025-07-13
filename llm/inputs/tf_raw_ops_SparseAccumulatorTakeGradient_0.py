
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseAccumulatorTakeGradient_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.constant("handle_1", dtype=tf.string)
    num_required = tf.constant(1, dtype=tf.int32)
    dtype = tf.float32
    name = "take_gradient_1"
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.constant("handle_2", dtype=tf.string)
    num_required = tf.constant(5, dtype=tf.int32)
    dtype = tf.int32
    name = "take_gradient_2"
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.constant("handle_3", dtype=tf.string)
    num_required = tf.constant(10, dtype=tf.int32)
    dtype = tf.float64
    name = "take_gradient_3"
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.constant("handle_4", dtype=tf.string)
    num_required = tf.constant(0, dtype=tf.int32)
    dtype = tf.int64
    name = "take_gradient_4"
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.constant("handle_5", dtype=tf.string)
    num_required = tf.constant(2, dtype=tf.int32)
    dtype = tf.complex64
    name = "take_gradient_5"
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.constant("handle_6", dtype=tf.string)
    num_required = tf.constant(7, dtype=tf.int32)
    dtype = tf.uint8
    name = "take_gradient_6"
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = tf.constant("handle_7", dtype=tf.string)
    num_required = tf.constant(3, dtype=tf.int32)
    dtype = tf.half
    name = "take_gradient_7"
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.constant("handle_8", dtype=tf.string)
    num_required = tf.constant(12, dtype=tf.int32)
    dtype = tf.bfloat16
    name = "take_gradient_8"
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.constant("handle_9", dtype=tf.string)
    num_required = tf.constant(4, dtype=tf.int32)
    dtype = tf.complex128
    name = "take_gradient_9"
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = tf.constant("handle_10", dtype=tf.string)
    num_required = tf.constant(8, dtype=tf.int32)
    dtype = tf.qint8
    name = "take_gradient_10"
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseAccumulatorTakeGradient"] = tf_raw_ops_SparseAccumulatorTakeGradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseAccumulatorTakeGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseAccumulatorTakeGradient'.")

check_valid('tf.raw_ops.SparseAccumulatorTakeGradient', generated_inputs['tf.raw_ops.SparseAccumulatorTakeGradient'], lib="tf", suffix=0)
