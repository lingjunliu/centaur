
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ordered_map_clear_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "dtypes": [tf.float32],
        "capacity": 0,
        "memory_limit": 0,
        "container": "",
        "shared_name": "",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "dtypes": [tf.int32, tf.float64],
        "capacity": 10,
        "memory_limit": 1024,
        "container": "my_container",
        "shared_name": "my_shared_name",
        "name": "ClearOp2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "dtypes": [tf.string],
        "capacity": 100,
        "memory_limit": 1024 * 1024,
        "container": "another_container",
        "shared_name": "another_shared_name",
        "name": "ClearOp3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "dtypes": [tf.bool, tf.int64, tf.float16],
        "capacity": 1,
        "memory_limit": 1,
        "container": "small_container",
        "shared_name": "small_shared_name",
        "name": "ClearOp4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "dtypes": [tf.complex64],
        "capacity": 50,
        "memory_limit": 50 * 1024,
        "container": "complex_container",
        "shared_name": "complex_shared_name",
        "name": "ClearOp5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "dtypes": [tf.uint8],
        "capacity": 128,
        "memory_limit": 2048,
        "container": "uint8_container",
        "shared_name": "uint8_shared_name",
        "name": "ClearOp6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "dtypes": [tf.resource],
        "capacity": 256,
        "memory_limit": 4096,
        "container": "resource_container",
        "shared_name": "resource_shared_name",
        "name": "ClearOp7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "dtypes": [tf.variant],
        "capacity": 0,
        "memory_limit": 0,
        "container": "variant_container",
        "shared_name": "variant_shared_name",
        "name": "ClearOp8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "dtypes": [tf.bfloat16],
        "capacity": 10,
        "memory_limit": 1000,
        "container": "bfloat16_container",
        "shared_name": "bfloat16_shared_name",
        "name": "ClearOp9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "dtypes": [tf.qint8],
        "capacity": 5,
        "memory_limit": 500,
        "container": "qint8_container",
        "shared_name": "qint8_shared_name",
        "name": "ClearOp10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.OrderedMapClear"] = tf_raw_ops_ordered_map_clear_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.OrderedMapClear' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.OrderedMapClear'.")

check_valid('tf.raw_ops.OrderedMapClear', generated_inputs['tf.raw_ops.OrderedMapClear'], lib="tf", suffix=0)
