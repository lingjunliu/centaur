
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_OrderedMapSize_inputs():
    list_of_inputs = []

    # Input 1
    dtypes = [tf.float32]
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

    # Input 2
    dtypes = [tf.int64, tf.string]
    capacity = 10
    memory_limit = 1024
    container = "my_container"
    shared_name = "my_shared_name"
    name = "MyOp"
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
    dtypes = [tf.bool, tf.int32, tf.float64]
    capacity = 100
    memory_limit = 1048576
    container = "container_123"
    shared_name = "shared_123"
    name = "Op_3"
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
    dtypes = [tf.complex64]
    capacity = 5
    memory_limit = 512
    container = "complex_container"
    shared_name = "complex_shared"
    name = "ComplexOp"
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
    dtypes = [tf.qint8, tf.quint8, tf.qint32]
    capacity = 20
    memory_limit = 2048
    container = "quantized_container"
    shared_name = "quantized_shared"
    name = "QuantizedOp"
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
    dtypes = [tf.resource, tf.variant]
    capacity = 50
    memory_limit = 524288
    container = "resource_container"
    shared_name = "resource_shared"
    name = "ResourceOp"
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
    dtypes = [tf.bfloat16]
    capacity = 2
    memory_limit = 256
    container = "bfloat_container"
    shared_name = "bfloat_shared"
    name = "BFloatOp"
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
    dtypes = [tf.uint8]
    capacity = 75
    memory_limit = 786432
    container = "uint8_container"
    shared_name = "uint8_shared"
    name = "UInt8Op"
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
    dtypes = [tf.int16, tf.uint16]
    capacity = 3
    memory_limit = 384
    container = "int16_container"
    shared_name = "int16_shared"
    name = "Int16Op"
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
    dtypes = [tf.int8]
    capacity = 120
    memory_limit = 12582912
    container = "int8_container"
    shared_name = "int8_shared"
    name = "Int8Op"
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
generated_inputs["tf.raw_ops.OrderedMapSize"] = tf_raw_ops_OrderedMapSize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.OrderedMapSize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.OrderedMapSize'.")

check_valid('tf.raw_ops.OrderedMapSize', generated_inputs['tf.raw_ops.OrderedMapSize'], lib="tf", suffix=0)
