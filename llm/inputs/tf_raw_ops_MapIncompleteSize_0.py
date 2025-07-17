
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
    name = None
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    dtypes = [tf.float32]
    capacity = 10
    memory_limit = 1024
    container = ""
    shared_name = ""
    name = "my_op"
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    dtypes = [tf.string]
    capacity = 1
    memory_limit = 1
    container = ""
    shared_name = ""
    name = None
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    dtypes = [tf.bool]
    capacity = 100
    memory_limit = 10000
    container = ""
    shared_name = ""
    name = "op4"
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    dtypes = [tf.uint8]
    capacity = 5
    memory_limit = 512
    container = ""
    shared_name = ""
    name = None
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    dtypes = [tf.int16]
    capacity = 20
    memory_limit = 2048
    container = ""
    shared_name = ""
    name = "op6"
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    dtypes = [tf.qint8]
    capacity = 0
    memory_limit = 0
    container = ""
    shared_name = ""
    name = None
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    dtypes = [tf.quint8]
    capacity = 1000
    memory_limit = 1000000
    container = ""
    shared_name = ""
    name = "my_large_op"
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    dtypes = [tf.resource]
    capacity = 2
    memory_limit = 128
    container = ""
    shared_name = ""
    name = None
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    dtypes = [tf.variant]
    capacity = 50
    memory_limit = 50000
    container = ""
    shared_name = ""
    name = "variant_op"
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    dtypes = [tf.int64]
    capacity = 256
    memory_limit = 262144
    container = ""
    shared_name = "shared_resource"
    name = None
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
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
