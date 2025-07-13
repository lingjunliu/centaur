
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_GetSessionTensor_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.constant("handle1", dtype=tf.string)
    dtype = np.float32
    name = "tensor1"
    input_dict = {"handle": handle, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.constant("handle2", dtype=tf.string)
    dtype = np.int32
    name = "tensor2"
    input_dict = {"handle": handle, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.constant("handle3", dtype=tf.string)
    dtype = np.bool_
    name = "tensor3"
    input_dict = {"handle": handle, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.constant("handle4", dtype=tf.string)
    dtype = np.float32
    name = "tensor4"
    input_dict = {"handle": handle, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.constant("handle5", dtype=tf.string)
    dtype = np.float64
    name = "tensor5"
    input_dict = {"handle": handle, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.constant("handle6", dtype=tf.string)
    dtype = np.int64
    name = "tensor6"
    input_dict = {"handle": handle, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = tf.constant("handle7", dtype=tf.string)
    dtype = np.complex64
    name = "tensor7"
    input_dict = {"handle": handle, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.constant("handle8", dtype=tf.string)
    dtype = np.complex128
    name = "tensor8"
    input_dict = {"handle": handle, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.constant("handle9", dtype=tf.string)
    dtype = np.int8
    name = "tensor9"
    input_dict = {"handle": handle, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = tf.constant("handle10", dtype=tf.string)
    dtype = np.uint8
    name = "tensor10"
    input_dict = {"handle": handle, "dtype": dtype, "name": name}
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
