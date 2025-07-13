
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_unstage_inputs():
    list_of_inputs = []

    # Input 1
    dtypes = [tf.float32]
    capacity = 0
    memory_limit = 0
    container = ""
    shared_name = ""
    name = None
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    dtypes = [tf.int32, tf.float64]
    capacity = 5
    memory_limit = 1024
    container = "test_container"
    shared_name = "test_shared_name"
    name = "unstage_op"
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    dtypes = [tf.string, tf.bool, tf.uint8]
    capacity = 10
    memory_limit = 2048
    container = "container_2"
    shared_name = "shared_name_2"
    name = None
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    dtypes = [tf.complex64]
    capacity = 1
    memory_limit = 512
    container = "complex_container"
    shared_name = "complex_shared"
    name = "complex_unstage"
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    dtypes = [tf.qint8, tf.quint8]
    capacity = 2
    memory_limit = 0
    container = ""
    shared_name = ""
    name = None
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    dtypes = [tf.resource]
    capacity = 100
    memory_limit = 10000
    container = "resource_container"
    shared_name = "resource_shared"
    name = "resource_unstage"
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    dtypes = [tf.variant]
    capacity = 0
    memory_limit = 0
    container = ""
    shared_name = "variant_shared"
    name = None
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    dtypes = [tf.float16, tf.bfloat16]
    capacity = 3
    memory_limit = 768
    container = "float_container"
    shared_name = ""
    name = "float_unstage"
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9
    dtypes = [tf.int64]
    capacity = 128
    memory_limit = 2048
    container = "int64_container"
    shared_name = "int64_shared"
    name = "int64_unstage"
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    dtypes = [tf.float32, tf.int32, tf.string]
    capacity = 7
    memory_limit = 4096
    container = "mixed_container"
    shared_name = "mixed_shared"
    name = None
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    dtypes = [tf.double]
    capacity = 1
    memory_limit = 128
    container = "double_container"
    shared_name = "double_shared"
    name = "double_name"
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    dtypes = [tf.complex128]
    capacity = 4
    memory_limit = 256
    container = "complex128_container"
    shared_name = "complex128_shared"
    name = "complex128_name"
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 13 - Empty dtypes
    dtypes = []
    capacity = 0
    memory_limit = 0
    container = ""
    shared_name = ""
    name = None
    input_dict = {"dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Unstage"] = tf_raw_ops_unstage_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Unstage' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Unstage'.")

check_valid('tf.raw_ops.Unstage', generated_inputs['tf.raw_ops.Unstage'], lib="tf", suffix=0)
