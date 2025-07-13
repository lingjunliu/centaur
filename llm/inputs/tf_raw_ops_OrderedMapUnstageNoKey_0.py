
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ordered_map_unstage_no_key_inputs():
    list_of_inputs = []

    # Input 1
    indices = np.array([1, 2, 3], dtype=np.int32)
    dtypes = [tf.float32]
    capacity = 10
    memory_limit = 1024
    container = "test_container1"
    shared_name = "test_shared_name1"
    name = "test_name1"

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
    indices = np.array([4, 5, 6, 7], dtype=np.int32)
    dtypes = [tf.int32, tf.float64]
    capacity = 5
    memory_limit = 2048
    container = "test_container2"
    shared_name = "test_shared_name2"
    name = "test_name2"

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

    # Input 3
    indices = np.array([8, 9], dtype=np.int32)
    dtypes = [tf.string]
    capacity = 0
    memory_limit = 0
    container = ""
    shared_name = ""
    name = "test_name3"

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

    # Input 4
    indices = np.array([], dtype=np.int32)
    dtypes = [tf.bool]
    capacity = 1
    memory_limit = 1
    container = "test_container4"
    shared_name = "test_shared_name4"
    name = "test_name4"

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

   # Input 5
    indices = np.array([10, 11, 12, 13, 14], dtype=np.int32)
    dtypes = [tf.int64, tf.float16, tf.complex64]
    capacity = 100
    memory_limit = 5000
    container = "container5"
    shared_name = "shared5"
    name = "name5"

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

    # Input 6
    indices = np.array([15], dtype=np.int32)
    dtypes = [tf.uint8]
    capacity = 2
    memory_limit = 512
    container = "test_container6"
    shared_name = "test_shared_name6"
    name = "test_name6"

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

    # Input 7
    indices = np.array([16,17,18], dtype=np.int32)
    dtypes = [tf.int8]
    capacity = 7
    memory_limit = 777
    container = "container7"
    shared_name = "shared7"
    name = "name7"

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

    # Input 8
    indices = np.array([19,20], dtype=np.int32)
    dtypes = [tf.bfloat16]
    capacity = 8
    memory_limit = 888
    container = "container8"
    shared_name = "shared8"
    name = "name8"

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

    # Input 9
    indices = np.array([21], dtype=np.int32)
    dtypes = [tf.float32, tf.int32]
    capacity = 9
    memory_limit = 999
    container = "container9"
    shared_name = "shared9"
    name = "name9"

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

    # Input 10
    indices = np.array([22,23,24,25], dtype=np.int32)
    dtypes = [tf.float32, tf.int32, tf.string]
    capacity = 10
    memory_limit = 10000
    container = "container10"
    shared_name = "shared10"
    name = "name10"

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
generated_inputs["tf.raw_ops.OrderedMapUnstageNoKey"] = tf_raw_ops_ordered_map_unstage_no_key_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.OrderedMapUnstageNoKey' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.OrderedMapUnstageNoKey'.")

check_valid('tf.raw_ops.OrderedMapUnstageNoKey', generated_inputs['tf.raw_ops.OrderedMapUnstageNoKey'], lib="tf", suffix=0)
