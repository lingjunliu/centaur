
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_LMDBReader_inputs():
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
    container = b"container_123"
    shared_name = b""
    name = b""
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    container = b""
    shared_name = b"shared_123"
    name = b""
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    container = b""
    shared_name = b""
    name = b"name_123"
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    container = b"long_container_name"
    shared_name = b"long_shared_name"
    name = b"long_name"
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    container = b"container_with_numbers_123"
    shared_name = b"shared_with_numbers_123"
    name = b"name_with_numbers_123"
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8
    container = b"special_chars_container!@#$"
    shared_name = b""
    name = b""
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    container = b""
    shared_name = b"special_chars_shared!@#$"
    name = b""
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    container = b""
    shared_name = b""
    name = b"special_chars_name!@#$"
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.LMDBReader"] = tf_raw_ops_LMDBReader_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.LMDBReader' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.LMDBReader'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.LMDBReader', generated_inputs['tf.raw_ops.LMDBReader'], lib="tf", suffix=0)
