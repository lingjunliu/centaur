
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MapIncompleteSize_inputs():
    list_of_inputs = []

    # Input 1
    dtypes = [tf.int32]
    capacity = 0
    memory_limit = 0
    container = ""
    shared_name = ""
    name = "map_incomplete_size_1"
    input_dict = {
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    dtypes = [tf.float32, tf.int64]
    capacity = 10
    memory_limit = 1024
    container = "my_container"
    shared_name = "my_shared_name"
    name = "map_incomplete_size_2"
    input_dict = {
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    dtypes = [tf.string]
    capacity = 100
    memory_limit = 2048
    container = "container3"
    shared_name = "shared3"
    name = None
    input_dict = {
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    dtypes = [tf.bool, tf.int8]
    capacity = 1
    memory_limit = 1
    container = "c4"
    shared_name = ""
    name = "map_incomplete_size_4"
    input_dict = {
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    dtypes = [tf.uint8, tf.int16, tf.float64]
    capacity = 5
    memory_limit = 512
    container = ""
    shared_name = "shared5"
    name = "map_incomplete_size_5"
    input_dict = {
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    dtypes = [tf.complex64]
    capacity = 0
    memory_limit = 0
    container = "container6"
    shared_name = ""
    name = "map_incomplete_size_6"
    input_dict = {
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    dtypes = [tf.resource, tf.variant]
    capacity = 2
    memory_limit = 200
    container = "container7"
    shared_name = "shared7"
    name = "map_incomplete_size_7"
    input_dict = {
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    dtypes = [tf.bfloat16]
    capacity = 128
    memory_limit = 4096
    container = ""
    shared_name = "shared_name_8"
    name = None

    input_dict = {
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    dtypes = [tf.uint32, tf.uint64]
    capacity = 64
    memory_limit = 2048
    container = "container_9"
    shared_name = ""
    name = "map_incomplete_size_9"

    input_dict = {
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    dtypes = [tf.half]
    capacity = 32
    memory_limit = 1024
    container = "container_10"
    shared_name = "shared_10"
    name = None

    input_dict = {
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
generated_inputs["tf.raw_ops.MapIncompleteSize"] = tf_raw_ops_MapIncompleteSize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MapIncompleteSize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MapIncompleteSize'.")

check_valid('tf.raw_ops.MapIncompleteSize', generated_inputs['tf.raw_ops.MapIncompleteSize'], lib="tf", suffix=0)
