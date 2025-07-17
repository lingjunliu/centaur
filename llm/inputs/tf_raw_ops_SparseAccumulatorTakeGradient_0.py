
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseAccumulatorTakeGradient_inputs():
    global generated_inputs
    # Initialize the dictionary before assigning the list
    if "tf.raw_ops.SparseAccumulatorTakeGradient" not in generated_inputs:
        generated_inputs["tf.raw_ops.SparseAccumulatorTakeGradient"] = []

    list_of_inputs = []

    # Input 1
    handle = tf.constant("accumulator_handle_1")
    num_required = tf.constant(1, dtype=tf.int32)
    dtype = tf.float32
    input_dict = {'handle': handle, 'num_required': num_required, 'dtype': dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.constant("accumulator_handle_2")
    num_required = tf.constant(5, dtype=tf.int32)
    dtype = tf.float64
    input_dict = {'handle': handle, 'num_required': num_required, 'dtype': dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.constant("accumulator_handle_3")
    num_required = tf.constant(10, dtype=tf.int32)
    dtype = tf.int32
    input_dict = {'handle': handle, 'num_required': num_required, 'dtype': dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.constant("accumulator_handle_4")
    num_required = tf.constant(2, dtype=tf.int32)
    dtype = tf.uint8
    input_dict = {'handle': handle, 'num_required': num_required, 'dtype': dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.constant("accumulator_handle_5")
    num_required = tf.constant(3, dtype=tf.int32)
    dtype = tf.int16
    input_dict = {'handle': handle, 'num_required': num_required, 'dtype': dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.constant("accumulator_handle_6")
    num_required = tf.constant(4, dtype=tf.int32)
    dtype = tf.int8
    input_dict = {'handle': handle, 'num_required': num_required, 'dtype': dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = tf.constant("accumulator_handle_7")
    num_required = tf.constant(1, dtype=tf.int32)
    dtype = tf.complex64
    input_dict = {'handle': handle, 'num_required': num_required, 'dtype': dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.constant("accumulator_handle_8")
    num_required = tf.constant(5, dtype=tf.int32)
    dtype = tf.int64
    input_dict = {'handle': handle, 'num_required': num_required, 'dtype': dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.constant("accumulator_handle_9")
    num_required = tf.constant(10, dtype=tf.int32)
    dtype = tf.half
    input_dict = {'handle': handle, 'num_required': num_required, 'dtype': dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = tf.constant("accumulator_handle_10")
    num_required = tf.constant(2, dtype=tf.int32)
    dtype = tf.complex128
    input_dict = {'handle': handle, 'num_required': num_required, 'dtype': dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    generated_inputs["tf.raw_ops.SparseAccumulatorTakeGradient"] = list_of_inputs

    return

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseAccumulatorTakeGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseAccumulatorTakeGradient'.")

check_valid('tf.raw_ops.SparseAccumulatorTakeGradient', generated_inputs['tf.raw_ops.SparseAccumulatorTakeGradient'], lib="tf", suffix=0)
