
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_WholeFileReader_inputs():
    list_of_inputs = []

    # Input 1
    container = b""
    shared_name = b""
    name = b"" if isinstance(None, type(None)) else "".encode('utf-8')

    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    container = b"my_container"
    shared_name = b"my_shared_name"
    name = b"" if isinstance(None, type(None)) else "".encode('utf-8')

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
    shared_name = b"unique_shared_name"
    name = b"" if isinstance(None, type(None)) else "".encode('utf-8')

    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    container = b"container_5"
    shared_name = b"shared_5"
    name = b"" if isinstance(None, type(None)) else "".encode('utf-8')

    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    container = b""
    shared_name = b""
    name = b"name_6"

    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    container = b"container_7"
    shared_name = b""
    name = b"" if isinstance(None, type(None)) else "".encode('utf-8')

    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    container = b""
    shared_name = b"shared_8"
    name = b"name_8"

    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    container = b"a_very_long_container_name"
    shared_name = b"a_very_long_shared_name"
    name = b"" if isinstance(None, type(None)) else "".encode('utf-8')

    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    container = b"container_10"
    shared_name = b"shared_10"
    name = b"name_10"

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
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.WholeFileReader' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.WholeFileReader'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.WholeFileReader', generated_inputs['tf.raw_ops.WholeFileReader'], lib="tf", suffix=0)
