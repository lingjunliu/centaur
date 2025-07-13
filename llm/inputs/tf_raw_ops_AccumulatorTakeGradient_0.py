
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_AccumulatorTakeGradient_inputs():
    list_of_inputs = []

    # Input 1
    handle = "test_handle_1"
    num_required = np.array(1, dtype=np.int32)
    dtype = tf.float32
    name = "take_gradient_1"
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = "test_handle_2"
    num_required = np.array(5, dtype=np.int32)
    dtype = tf.float64
    name = "take_gradient_2"
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = "test_handle_3"
    num_required = np.array(10, dtype=np.int32)
    dtype = tf.int32
    name = "take_gradient_3"
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = "test_handle_4"
    num_required = np.array(0, dtype=np.int32)
    dtype = tf.complex64
    name = "take_gradient_4"
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = "test_handle_5"
    num_required = np.array(1, dtype=np.int32)
    dtype = tf.half
    name = "take_gradient_5"
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    handle = "test_handle_6"
    num_required = np.array(2, dtype=np.int32)
    dtype = tf.bfloat16
    name = "take_gradient_6"
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = "test_handle_7"
    num_required = np.array(1, dtype=np.int32)
    dtype = tf.qint8
    name = "take_gradient_7"
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = "test_handle_8"
    num_required = np.array(1, dtype=np.int32)
    dtype = tf.quint8
    name = "take_gradient_8"
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = "test_handle_9"
    num_required = np.array(5, dtype=np.int32)
    dtype = tf.qint32
    name = "take_gradient_9"
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = "test_handle_10"
    num_required = np.array(10, dtype=np.int32)
    dtype = tf.uint8
    name = "take_gradient_10"
    input_dict = {"handle": handle, "num_required": num_required, "dtype": dtype, "name": name}
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
