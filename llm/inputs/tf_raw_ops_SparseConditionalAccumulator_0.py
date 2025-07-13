
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sparse_conditional_accumulator_inputs():
    list_of_inputs = []

    # Input 1
    dtype = tf.float32
    shape = [2, 3]
    container = ""
    shared_name = ""
    reduction_type = "MEAN"
    name = ""
    input_dict = {
        "dtype": dtype,
        "shape": shape,
        "container": container,
        "shared_name": shared_name,
        "reduction_type": reduction_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    dtype = tf.int32
    shape = [5]
    container = "my_container"
    shared_name = "my_accumulator"
    reduction_type = "SUM"
    name = "accumulator_1"
    input_dict = {
        "dtype": dtype,
        "shape": shape,
        "container": container,
        "shared_name": shared_name,
        "reduction_type": reduction_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    dtype = tf.float64
    shape = [1, 4, 2]
    container = ""
    shared_name = "another_accumulator"
    reduction_type = "MEAN"
    name = "accumulator_2"
    input_dict = {
        "dtype": dtype,
        "shape": shape,
        "container": container,
        "shared_name": shared_name,
        "reduction_type": reduction_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    dtype = tf.int64
    shape = [10, 10]
    container = "container_2"
    shared_name = ""
    reduction_type = "SUM"
    name = ""
    input_dict = {
        "dtype": dtype,
        "shape": shape,
        "container": container,
        "shared_name": shared_name,
        "reduction_type": reduction_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    dtype = tf.complex64
    shape = [3, 3, 3]
    container = ""
    shared_name = "complex_accumulator"
    reduction_type = "MEAN"
    name = "complex_acc"
    input_dict = {
        "dtype": dtype,
        "shape": shape,
        "container": container,
        "shared_name": shared_name,
        "reduction_type": reduction_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    dtype = tf.bfloat16
    shape = [4]
    container = "bfloat_container"
    shared_name = "bfloat_acc"
    reduction_type = "SUM"
    name = "bfloat16_accumulator"
    input_dict = {
        "dtype": dtype,
        "shape": shape,
        "container": container,
        "shared_name": shared_name,
        "reduction_type": reduction_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    dtype = tf.half
    shape = [2, 2]
    container = ""
    shared_name = ""
    reduction_type = "MEAN"
    name = ""
    input_dict = {
        "dtype": dtype,
        "shape": shape,
        "container": container,
        "shared_name": shared_name,
        "reduction_type": reduction_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    dtype = tf.uint8
    shape = [8, 8]
    container = "uint8_container"
    shared_name = "uint8_acc"
    reduction_type = "SUM"
    name = "uint8_accumulator"
    input_dict = {
        "dtype": dtype,
        "shape": shape,
        "container": container,
        "shared_name": shared_name,
        "reduction_type": reduction_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    dtype = tf.qint32
    shape = [16]
    container = ""
    shared_name = ""
    reduction_type = "MEAN"
    name = ""
    input_dict = {
        "dtype": dtype,
        "shape": shape,
        "container": container,
        "shared_name": shared_name,
        "reduction_type": reduction_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    dtype = tf.uint64
    shape = [1, 1, 1, 1]
    container = "uint64_container"
    shared_name = "uint64_acc"
    reduction_type = "SUM"
    name = "uint64_accumulator"
    input_dict = {
        "dtype": dtype,
        "shape": shape,
        "container": container,
        "shared_name": shared_name,
        "reduction_type": reduction_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseConditionalAccumulator"] = tf_raw_ops_sparse_conditional_accumulator_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseConditionalAccumulator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseConditionalAccumulator'.")

check_valid('tf.raw_ops.SparseConditionalAccumulator', generated_inputs['tf.raw_ops.SparseConditionalAccumulator'], lib="tf", suffix=0)
