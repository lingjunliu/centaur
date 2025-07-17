
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_variable_inputs():
    list_of_inputs = []

    # Input 1
    shape = [2, 3]
    dtype = tf.float32
    container = ""
    shared_name = ""
    name = "variable_1"

    input_dict = {
        "shape": shape,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = [5]
    dtype = tf.int32
    container = "container_2"
    shared_name = "shared_2"
    name = "variable_2"

    input_dict = {
        "shape": shape,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = [1, 4, 2]
    dtype = tf.complex64
    container = ""
    shared_name = "shared_3"
    name = "variable_3"

    input_dict = {
        "shape": shape,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = []
    dtype = tf.bool
    container = "container_4"
    shared_name = ""
    name = "variable_4"

    input_dict = {
        "shape": shape,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = [2, 2]
    dtype = tf.float64
    container = ""
    shared_name = ""
    name = "variable_5"

    input_dict = {
        "shape": shape,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = [1]
    dtype = tf.string
    container = "container_6"
    shared_name = "shared_6"
    name = "variable_6"

    input_dict = {
        "shape": shape,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    shape = [2,2]
    dtype = tf.uint8
    container = "container_7"
    shared_name = ""
    name = "variable_7"

    input_dict = {
        "shape": shape,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = [3]
    dtype = tf.int64
    container = ""
    shared_name = "shared_8"
    name = "variable_8"

    input_dict = {
        "shape": shape,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape = [1,1]
    dtype = tf.bfloat16
    container = "container_9"
    shared_name = ""
    name = "variable_9"

    input_dict = {
        "shape": shape,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = [4]
    dtype = tf.float16
    container = ""
    shared_name = "shared_10"
    name = "variable_10"

    input_dict = {
        "shape": shape,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Variable"] = tf_raw_ops_variable_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Variable' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Variable'.")

check_valid('tf.raw_ops.Variable', generated_inputs['tf.raw_ops.Variable'], lib="tf", suffix=0)
