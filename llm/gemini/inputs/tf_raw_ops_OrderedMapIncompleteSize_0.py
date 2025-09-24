
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_OrderedMapIncompleteSize_inputs():
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
    dtypes = [tf.int64]
    capacity = 100
    memory_limit = 1024 * 1024
    container = ""
    shared_name = ""
    name = "op2"
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    dtypes = [tf.string]
    capacity = 1
    memory_limit = 1
    container = ""
    shared_name = ""
    name = "c"
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    dtypes = [tf.bool]
    capacity = 5
    memory_limit = 512
    container = ""
    shared_name = ""
    name = "bool_op"
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    dtypes = [tf.complex64]
    capacity = 128
    memory_limit = 64 * 1024
    container = ""
    shared_name = ""
    name = "complex_op"
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    dtypes = [tf.qint8]
    capacity = 256
    memory_limit = 128 * 1024
    container = ""
    shared_name = ""
    name = "quantized_op"
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    dtypes = [tf.resource]
    capacity = 0
    memory_limit = 0
    container = ""
    shared_name = ""
    name = "resource_op"
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    dtypes = [tf.variant]
    capacity = 1024
    memory_limit = 512 * 1024
    container = ""
    shared_name = ""
    name = "variant_op"
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    dtypes = [tf.bfloat16]
    capacity = 2048
    memory_limit = 256 * 1024
    container = ""
    shared_name = ""
    name = "bfloat16_op"
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.OrderedMapIncompleteSize"] = tf_raw_ops_OrderedMapIncompleteSize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.OrderedMapIncompleteSize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.OrderedMapIncompleteSize'.")

check_valid('tf.raw_ops.OrderedMapIncompleteSize', generated_inputs['tf.raw_ops.OrderedMapIncompleteSize'], lib="tf", suffix=0)
