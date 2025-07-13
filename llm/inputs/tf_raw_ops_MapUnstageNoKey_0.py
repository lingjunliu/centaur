
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MapUnstageNoKey_inputs():
    list_of_inputs = []

    # Input 1
    indices = np.array([1, 2, 3], dtype=np.int32)
    dtypes = [tf.float32]
    capacity = 10
    memory_limit = 100
    container = "test_container"
    shared_name = "test_shared_name"
    name = "test_name"
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
    capacity = 0
    memory_limit = 0
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

    # Input 3
    indices = np.array([8, 9], dtype=np.int32)
    dtypes = [tf.string]
    capacity = 5
    memory_limit = 50
    container = "container2"
    shared_name = "shared2"
    name = "name2"
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
    container = "c"
    shared_name = "s"
    name = "n"
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
    indices = np.array([10,11,12,13,14], dtype=np.int32)
    dtypes = [tf.uint8]
    capacity = 12
    memory_limit = 120
    container = "test_container3"
    shared_name = "test_shared_name3"
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

    # Input 6
    indices = np.array([[15,16],[17,18]], dtype=np.int32)
    dtypes = [tf.bfloat16]
    capacity = 2
    memory_limit = 20
    container = "container4"
    shared_name = "shared4"
    name = "name4"
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
    indices = np.array([19], dtype=np.int32)
    dtypes = [tf.complex64]
    capacity = 15
    memory_limit = 150
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

    # Input 8
    indices = np.array([20,21,22], dtype=np.int32)
    dtypes = [tf.variant]
    capacity = 0
    memory_limit = 0
    container = "container6"
    shared_name = "shared6"
    name = "name6"
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
    indices = np.array([23,24], dtype=np.int32)
    dtypes = [tf.qint8]
    capacity = 7
    memory_limit = 70
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

    # Input 10
    indices = np.array([25], dtype=np.int32)
    dtypes = [tf.quint16]
    capacity = 3
    memory_limit = 30
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
