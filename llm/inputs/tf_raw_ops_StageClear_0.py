
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_raw_ops_stageclear_inputs():
    list_of_inputs = []

    # Input 1
    dtypes = [tf.float32.as_numpy_dtype]
    capacity = 10
    memory_limit = 1024
    container = "testcontainer1"
    shared_name = "testsharedname1"
    name = "testname1"
    input_dict = {
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    dtypes = [tf.int32.as_numpy_dtype]
    capacity = 0
    memory_limit = 0
    container = ""
    shared_name = ""
    name = None
    input_dict = {
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    dtypes = [tf.string.as_numpy_dtype]
    capacity = 1
    memory_limit = 1
    container = "container3"
    shared_name = "shared3"
    name = "name3"
    input_dict = {
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    dtypes = [tf.bool.as_numpy_dtype]
    capacity = 100
    memory_limit = 10000
    container = "container4"
    shared_name = "shared4"
    name = "name4"
    input_dict = {
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    dtypes = [tf.uint8.as_numpy_dtype]
    capacity = 2
    memory_limit = 2048
    container = "container5"
    shared_name = "sharedname5"
    name = "name5"
    input_dict = {
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    dtypes = [tf.float16.as_numpy_dtype]
    capacity = 5
    memory_limit = 512
    container = "resourcecontainer"
    shared_name = "resourcesharedname"
    name = "resourcename"
    input_dict = {
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    dtypes = [tf.int8.as_numpy_dtype]
    capacity = 7
    memory_limit = 777
    container = "variantcontainer"
    shared_name = "variantshared"
    name = "variantname"
    input_dict = {
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    dtypes = [tf.bfloat16.as_numpy_dtype]
    capacity = 8
    memory_limit = 8888
    container = "bfloat16container"
    shared_name = "bfloat16shared"
    name = "bfloat16name"
    input_dict = {
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    dtypes = [tf.qint32.as_numpy_dtype]
    capacity = 9
    memory_limit = 999
    container = "qint8container"
    shared_name = "qint8shared"
    name = "qint8name"
    input_dict = {
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10
    dtypes = [tf.quint16.as_numpy_dtype]
    capacity = 10
    memory_limit = 1000
    container = "quint8container"
    shared_name = "quint8shared"
    name = "quint8name"
    input_dict = {
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.StageClear"] = tf_raw_ops_stageclear_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.StageClear' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StageClear'.")

check_valid('tf.raw_ops.StageClear', generated_inputs['tf.raw_ops.StageClear'], lib="tf", suffix=0)
