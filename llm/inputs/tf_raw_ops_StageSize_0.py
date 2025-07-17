
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_raw_ops_stage_size_inputs():
    list_of_inputs = []

    # Input 1
    dtypes = [tf.float32]
    capacity = 10
    memory_limit = 1024
    container = "container1"
    shared_name = "shared1"
    name = "stage_size1"
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
    dtypes = [tf.int32]
    capacity = 0
    memory_limit = 0
    container = ""
    shared_name = ""
    name = "stage_size2"
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
    capacity = 5
    memory_limit = 512
    container = "container2"
    shared_name = "shared2"
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
    dtypes = [tf.bool]
    capacity = 100
    memory_limit = 10000
    container = "container3"
    shared_name = "shared3"
    name = "stage_size4"
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
    dtypes = [tf.float32]
    capacity = 1
    memory_limit = 1
    container = "container4"
    shared_name = "shared4"
    name = "stage_size5"
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
    dtypes = [tf.uint8]
    capacity = 20
    memory_limit = 2048
    container = "container5"
    shared_name = "shared5"
    name = "stage_size6"
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
    dtypes = [tf.complex64]
    capacity = 30
    memory_limit = 3072
    container = "container6"
    shared_name = "shared6"
    name = "stage_size7"
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
    dtypes = [tf.resource]
    capacity = 40
    memory_limit = 4096
    container = "container7"
    shared_name = "shared7"
    name = "stage_size8"
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
    dtypes = [tf.variant]
    capacity = 50
    memory_limit = 5120
    container = "container8"
    shared_name = "shared8"
    name = "stage_size9"
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
    dtypes = [tf.qint8]
    capacity = 60
    memory_limit = 6144
    container = "container9"
    shared_name = "shared9"
    name = "stage_size10"
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
generated_inputs["tf.raw_ops.StageSize"] = tf_raw_ops_stage_size_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.StageSize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StageSize'.")

check_valid('tf.raw_ops.StageSize', generated_inputs['tf.raw_ops.StageSize'], lib="tf", suffix=0)
