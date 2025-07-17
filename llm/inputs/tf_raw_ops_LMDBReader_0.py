
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_raw_ops_lmdb_reader_inputs():
    list_of_inputs = []

    # Input 1
    container = np.array(b"", dtype=np.dtype('S'))
    shared_name = np.array(b"", dtype=np.dtype('S'))
    name = np.array(b"", dtype=np.dtype('S'))
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    container = np.array(b"my_container", dtype=np.dtype('S'))
    shared_name = np.array(b"my_shared_name", dtype=np.dtype('S'))
    name = np.array(b"my_op_name", dtype=np.dtype('S'))
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    container = np.array(b"another_container", dtype=np.dtype('S'))
    shared_name = np.array(b"", dtype=np.dtype('S'))
    name = np.array(b"another_op_name", dtype=np.dtype('S'))
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    container = np.array(b"", dtype=np.dtype('S'))
    shared_name = np.array(b"yet_another_shared_name", dtype=np.dtype('S'))
    name = np.array(b"", dtype=np.dtype('S'))
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    container = np.array(b"a_very_long_container_name", dtype=np.dtype('S'))
    shared_name = np.array(b"a_very_long_shared_name", dtype=np.dtype('S'))
    name = np.array(b"a_very_long_op_name", dtype=np.dtype('S'))
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    container = np.array(b"container_6", dtype=np.dtype('S'))
    shared_name = np.array(b"", dtype=np.dtype('S'))
    name = np.array(b"", dtype=np.dtype('S'))
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    container = np.array(b"", dtype=np.dtype('S'))
    shared_name = np.array(b"shared_name_7", dtype=np.dtype('S'))
    name = np.array(b"name_7", dtype=np.dtype('S'))
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    container = np.array(b"container_8", dtype=np.dtype('S'))
    shared_name = np.array(b"shared_name_8", dtype=np.dtype('S'))
    name = np.array(b"", dtype=np.dtype('S'))
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    container = np.array(b"9", dtype=np.dtype('S'))
    shared_name = np.array(b"", dtype=np.dtype('S'))
    name = np.array(b"9", dtype=np.dtype('S'))
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    container = np.array(b"", dtype=np.dtype('S'))
    shared_name = np.array(b"10", dtype=np.dtype('S'))
    name = np.array(b"", dtype=np.dtype('S'))
    input_dict = {
        "container": container,
        "shared_name": shared_name,
        "name": name
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
