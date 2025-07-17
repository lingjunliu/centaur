
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_mapclear_inputs():
    list_of_inputs = []

    # Input 1
    dtypes = [tf.float32]
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
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    dtypes = [tf.int32]
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
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    dtypes = [tf.string]
    capacity = 100
    memory_limit = 5000
    container = "container3"
    shared_name = "shared3"
    name = "name3"

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
    capacity = 1
    memory_limit = 1
    container = "boolcontainer"
    shared_name = "boolshared"
    name = "boolname"

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
    dtypes = [tf.float16]
    capacity = 20
    memory_limit = 2048
    container = "mixedcontainer"
    shared_name = "mixedshared"
    name = "mixedname"

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
    capacity = 5
    memory_limit = 100
    container = "complexcontainer"
    shared_name = "complexshared"
    name = "complexname"

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
    dtypes = [tf.resource]
    capacity = 1000
    memory_limit = 1000000
    container = "resourcecontainer"
    shared_name = "resourceshared"
    name = "resourcename"

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
    dtypes = [tf.variant]
    capacity = 2
    memory_limit = 20
    container = "variantcontainer"
    shared_name = "variantshared"
    name = "variantname"

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
    dtypes = [tf.qint8]
    capacity = 7
    memory_limit = 7777
    container = "quantizedcontainer"
    shared_name = "quantizedshared"
    name = "quantizedname"

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
    dtypes = [tf.bfloat16]
    capacity = 33
    memory_limit = 3333
    container = "bfloatcontainer"
    shared_name = "bfloatshared"
    name = "bfloatname"

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
generated_inputs["tf.raw_ops.MapClear"] = tf_raw_ops_mapclear_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MapClear' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MapClear'.")

check_valid('tf.raw_ops.MapClear', generated_inputs['tf.raw_ops.MapClear'], lib="tf", suffix=0)
