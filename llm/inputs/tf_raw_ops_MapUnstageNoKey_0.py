
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MapUnstageNoKey_inputs():
    list_of_inputs = []

    # Input 1
    indices = np.array([1], dtype=np.int32)
    dtypes = [tf.float32]
    capacity = 10
    memory_limit = 1024
    container = "testcontainer"
    shared_name = "testsharedname"
    name = "testname"

    input_dict = {
        "indices": indices,
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = np.array([4], dtype=np.int32)
    dtypes = [tf.int32]
    capacity = 5
    memory_limit = 512
    container = ""
    shared_name = ""
    name = None

    input_dict = {
        "indices": indices,
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MapUnstageNoKey"] = tf_raw_ops_MapUnstageNoKey_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MapUnstageNoKey' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MapUnstageNoKey'.")

check_valid('tf.raw_ops.MapUnstageNoKey', generated_inputs['tf.raw_ops.MapUnstageNoKey'], lib="tf", suffix=0)
