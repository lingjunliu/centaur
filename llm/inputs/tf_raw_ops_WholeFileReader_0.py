
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_raw_ops_WholeFileReader_inputs():
    list_of_inputs = []

    # Input 1
    container = b""
    shared_name = b""
    name = b""
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    container = b"my_container"
    shared_name = b"my_shared_name"
    name = b"my_name"
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    container = b"another_container"
    shared_name = b""
    name = b"another_name"
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    container = b""
    shared_name = b"another_shared_name"
    name = b""
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    container = b"a_long_container_name"
    shared_name = b"a_long_shared_name"
    name = b"a_long_name"
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    container = b"container_with_numbers_123"
    shared_name = b"shared_name_with_numbers_456"
    name = b"name_with_numbers_789"
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    container = b"container_with_symbols_!@#$"
    shared_name = b"shared_name_with_symbols_%^&*"
    name = b"name_with_symbols_()_+=-"
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    container = b""
    shared_name = b"shared_name_with_empty_container"
    name = b"name_with_empty_container"
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    container = b"container_with_empty_shared_name"
    shared_name = b""
    name = b"name_with_empty_shared_name"
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    container = b"container_with_empty_name"
    shared_name = b"shared_name_with_empty_name"
    name = b""
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.WholeFileReader"] = tf_raw_ops_WholeFileReader_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.WholeFileReader' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.WholeFileReader'.")

check_valid('tf.raw_ops.WholeFileReader', generated_inputs['tf.raw_ops.WholeFileReader'], lib="tf", suffix=0)
