
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_OrderedMapClear_inputs():
    list_of_inputs = []

    # Input 1
    dtypes = [tf.float32]
    capacity = 10
    memory_limit = 1024
    container = "container1"
    shared_name = "shared1"
    name = "op1"

    input_dict = {
        "dtypes": [d.name for d in dtypes if isinstance(d, type)],
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    dtypes = [tf.int32, tf.float64]
    capacity = 0
    memory_limit = 0
    container = ""
    shared_name = ""
    name = None

    input_dict = {
        "dtypes": [d.name for d in dtypes if isinstance(d, type)],
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
    container = "container2"
    shared_name = "shared2"
    name = "op3"

    input_dict = {
        "dtypes": [d.name for d in dtypes if isinstance(d, type)],
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 4
    dtypes = [tf.bool]
    capacity = 5
    memory_limit = 512
    container = "container3"
    shared_name = "shared3"
    name = "op4"

    input_dict = {
        "dtypes": [d.name for d in dtypes if isinstance(d, type)],
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    dtypes = [tf.uint8, tf.int16, tf.float16]
    capacity = 20
    memory_limit = 4096
    container = "container4"
    shared_name = "shared4"
    name = "op5"

    input_dict = {
        "dtypes": [d.name for d in dtypes if isinstance(d, type)],
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    dtypes = [tf.complex64]
    capacity = 1
    memory_limit = 128
    container = "container5"
    shared_name = "shared5"
    name = "op6"

    input_dict = {
        "dtypes": [d.name for d in dtypes if isinstance(d, type)],
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    dtypes = [tf.int32]
    capacity = 7
    memory_limit = 777
    container = "container6"
    shared_name = "shared6"
    name = "op7"

    input_dict = {
        "dtypes": [d.name for d in dtypes if isinstance(d, type)],
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    dtypes = [tf.qint8]
    capacity = 8
    memory_limit = 888
    container = "container7"
    shared_name = "shared7"
    name = "op8"

    input_dict = {
        "dtypes": [d.name for d in dtypes if isinstance(d, type)],
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    dtypes = [tf.quint8]
    capacity = 9
    memory_limit = 999
    container = "container8"
    shared_name = "shared8"
    name = "op9"

    input_dict = {
        "dtypes": [d.name for d in dtypes if isinstance(d, type)],
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    dtypes = [tf.bfloat16]
    capacity = 11
    memory_limit = 1111
    container = "container9"
    shared_name = "shared9"
    name = "op10"

    input_dict = {
        "dtypes": [d.name for d in dtypes if isinstance(d, type)],
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.OrderedMapClear"] = tf_raw_ops_OrderedMapClear_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.OrderedMapClear' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.OrderedMapClear'.")

check_valid('tf.raw_ops.OrderedMapClear', generated_inputs['tf.raw_ops.OrderedMapClear'], lib="tf", suffix=0)
