
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_immutable_const_inputs():
    list_of_inputs = []

    # Input 1
    dtype = tf.float32
    shape = [2, 3]
    memory_region_name = ""
    input_dict = {
        "dtype": dtype,
        "shape": shape,
        "memory_region_name": memory_region_name,
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    dtype = tf.int64
    shape = [5]
    memory_region_name = ""
    input_dict = {
        "dtype": dtype,
        "shape": shape,
        "memory_region_name": memory_region_name,
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    dtype = tf.bool
    shape = [1, 4, 2]
    memory_region_name = ""
    input_dict = {
        "dtype": dtype,
        "shape": shape,
        "memory_region_name": memory_region_name,
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    dtype = tf.complex64
    shape = [2, 2, 2, 2]
    memory_region_name = ""
    input_dict = {
        "dtype": dtype,
        "shape": shape,
        "memory_region_name": memory_region_name,
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    dtype = tf.string
    shape = [1]
    memory_region_name = ""
    input_dict = {
        "dtype": dtype,
        "shape": shape,
        "memory_region_name": memory_region_name,
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    dtype = tf.uint8
    shape = [10]
    memory_region_name = ""
    input_dict = {
        "dtype": dtype,
        "shape": shape,
        "memory_region_name": memory_region_name,
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    dtype = tf.qint8
    shape = [3, 3]
    memory_region_name = ""
    input_dict = {
        "dtype": dtype,
        "shape": shape,
        "memory_region_name": memory_region_name,
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    dtype = tf.resource
    shape = []
    memory_region_name = ""
    input_dict = {
        "dtype": dtype,
        "shape": shape,
        "memory_region_name": memory_region_name,
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    dtype = tf.variant
    shape = [1, 1, 1, 1, 1]
    memory_region_name = ""
    input_dict = {
        "dtype": dtype,
        "shape": shape,
        "memory_region_name": memory_region_name,
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    dtype = tf.bfloat16
    shape = [7, 1]
    memory_region_name = ""
    input_dict = {
        "dtype": dtype,
        "shape": shape,
        "memory_region_name": memory_region_name,
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ImmutableConst"] = tf_raw_ops_immutable_const_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ImmutableConst' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ImmutableConst'.")

check_valid('tf.raw_ops.ImmutableConst', generated_inputs['tf.raw_ops.ImmutableConst'], lib="tf", suffix=0)
