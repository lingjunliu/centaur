
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_StageClear_inputs():
    list_of_inputs = []

    # Input 1
    dtypes = [tf.float32.as_numpy_dtype]
    capacity = 0
    memory_limit = 0
    container = ""
    shared_name = ""
    name = None
    input_dict = {'dtypes': dtypes, 'capacity': capacity, 'memory_limit': memory_limit, 'container': container, 'shared_name': shared_name, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    dtypes = [tf.int32.as_numpy_dtype]
    capacity = 10
    memory_limit = 1024
    container = "test_container"
    shared_name = "test_shared_name"
    name = "StageClear_Op"
    input_dict = {'dtypes': dtypes, 'capacity': capacity, 'memory_limit': memory_limit, 'container': container, 'shared_name': shared_name, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    dtypes = [tf.string.as_numpy_dtype]
    capacity = 1
    memory_limit = 1
    container = "container_1"
    shared_name = "shared_name_1"
    name = None
    input_dict = {'dtypes': dtypes, 'capacity': capacity, 'memory_limit': memory_limit, 'container': container, 'shared_name': shared_name, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    dtypes = [tf.bool.as_numpy_dtype]
    capacity = 100
    memory_limit = 512
    container = "container_2"
    shared_name = "shared_name_2"
    name = "clear_stage"
    input_dict = {'dtypes': dtypes, 'capacity': capacity, 'memory_limit': memory_limit, 'container': container, 'shared_name': shared_name, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    dtypes = [tf.uint8.as_numpy_dtype]
    capacity = 5
    memory_limit = 2048
    container = ""
    shared_name = ""
    name = None
    input_dict = {'dtypes': dtypes, 'capacity': capacity, 'memory_limit': memory_limit, 'container': container, 'shared_name': shared_name, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    dtypes = [tf.float16.as_numpy_dtype]
    capacity = 0
    memory_limit = 0
    container = "float16_complex64"
    shared_name = "float16_complex64"
    name = "clear_stage_float16_complex64"
    input_dict = {'dtypes': dtypes, 'capacity': capacity, 'memory_limit': memory_limit, 'container': container, 'shared_name': shared_name, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    dtypes = [tf.int64.as_numpy_dtype]
    capacity = 2
    memory_limit = 4096
    container = "resource_container"
    shared_name = "resource_shared_name"
    name = None
    input_dict = {'dtypes': dtypes, 'capacity': capacity, 'memory_limit': memory_limit, 'container': container, 'shared_name': shared_name, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    dtypes = [tf.float64.as_numpy_dtype]
    capacity = 1000
    memory_limit = 65536
    container = "variant_container"
    shared_name = "variant_shared_name"
    name = "clear_stage_variant"
    input_dict = {'dtypes': dtypes, 'capacity': capacity, 'memory_limit': memory_limit, 'container': container, 'shared_name': shared_name, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    dtypes = [tf.qint8.as_numpy_dtype]
    capacity = 128
    memory_limit = 16384
    container = ""
    shared_name = ""
    name = None
    input_dict = {'dtypes': dtypes, 'capacity': capacity, 'memory_limit': memory_limit, 'container': container, 'shared_name': shared_name, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    dtypes = [tf.bfloat16.as_numpy_dtype]
    capacity = 64
    memory_limit = 8192
    container = "bfloat16_container"
    shared_name = "bfloat16_shared_name"
    name = "clear_stage_bfloat16"
    input_dict = {'dtypes': dtypes, 'capacity': capacity, 'memory_limit': memory_limit, 'container': container, 'shared_name': shared_name, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.StageClear"] = tf_raw_ops_StageClear_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.StageClear' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StageClear'.")

check_valid('tf.raw_ops.StageClear', generated_inputs['tf.raw_ops.StageClear'], lib="tf", suffix=0)
