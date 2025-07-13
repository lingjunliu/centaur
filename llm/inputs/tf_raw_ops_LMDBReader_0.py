
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_raw_ops_lmdb_reader_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "container": "".encode('utf-8'),
        "shared_name": "".encode('utf-8'),
        "name": "".encode('utf-8')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "container": "my_container".encode('utf-8'),
        "shared_name": "my_shared_name".encode('utf-8'),
        "name": "my_name".encode('utf-8')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "container": "another_container".encode('utf-8'),
        "shared_name": "".encode('utf-8'),
        "name": "".encode('utf-8')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "container": "".encode('utf-8'),
        "shared_name": "another_shared_name".encode('utf-8'),
        "name": "".encode('utf-8')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "container": "".encode('utf-8'),
        "shared_name": "".encode('utf-8'),
        "name": "another_name".encode('utf-8')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "container": "container_1".encode('utf-8'),
        "shared_name": "shared_name_1".encode('utf-8'),
        "name": "name_1".encode('utf-8')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "container": "container_2".encode('utf-8'),
        "shared_name": "".encode('utf-8'),
        "name": "name_2".encode('utf-8')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "container": "".encode('utf-8'),
        "shared_name": "shared_name_2".encode('utf-8'),
        "name": "name_3".encode('utf-8')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "container": "container_3".encode('utf-8'),
        "shared_name": "shared_name_3".encode('utf-8'),
        "name": "".encode('utf-8')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10
    input_dict = {
        "container": "very_long_container_name".encode('utf-8'),
        "shared_name": "very_long_shared_name".encode('utf-8'),
        "name": "very_long_name".encode('utf-8')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.LMDBReader"] = tf_raw_ops_lmdb_reader_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.LMDBReader' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.LMDBReader'.")

check_valid('tf.raw_ops.LMDBReader', generated_inputs['tf.raw_ops.LMDBReader'], lib="tf", suffix=0)
