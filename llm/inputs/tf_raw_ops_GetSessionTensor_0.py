
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_GetSessionTensor_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.constant("tensor_handle_1", dtype=tf.string)
    dtype = tf.float32

    input_dict = {
        "handle": handle,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.constant("tensor_handle_2", dtype=tf.string)
    dtype = tf.int32

    input_dict = {
        "handle": handle,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.constant("tensor_handle_3", dtype=tf.string)
    dtype = tf.bool

    input_dict = {
        "handle": handle,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.constant("tensor_handle_4", dtype=tf.string)
    dtype = tf.float64

    input_dict = {
        "handle": handle,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    handle = tf.constant("tensor_handle_5", dtype=tf.string)
    dtype = tf.int64

    input_dict = {
        "handle": handle,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.constant("tensor_handle_6", dtype=tf.string)
    dtype = tf.uint8

    input_dict = {
        "handle": handle,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = tf.constant("tensor_handle_7", dtype=tf.string)
    dtype = tf.complex64

    input_dict = {
        "handle": handle,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.constant("tensor_handle_8", dtype=tf.string)
    dtype = tf.complex128

    input_dict = {
        "handle": handle,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.constant("tensor_handle_9", dtype=tf.string)
    dtype = tf.qint8

    input_dict = {
        "handle": handle,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = tf.constant("tensor_handle_10", dtype=tf.string)
    dtype = tf.quint8

    input_dict = {
        "handle": handle,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.GetSessionTensor"] = tf_raw_ops_GetSessionTensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.GetSessionTensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.GetSessionTensor'.")

check_valid('tf.raw_ops.GetSessionTensor', generated_inputs['tf.raw_ops.GetSessionTensor'], lib="tf", suffix=0)
