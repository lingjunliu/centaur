
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ordered_map_unstage_inputs():
    list_of_inputs = []

    # Input 1
    key = np.array(1, dtype=np.int64)
    indices = np.array([0], dtype=np.int32)
    dtypes = [tf.float32]
    capacity = 0
    memory_limit = 0
    container = ""
    shared_name = ""
    name = None

    input_dict = {
        "key": key,
        "indices": indices,
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    key = np.array(2, dtype=np.int64)
    indices = np.array([1, 2], dtype=np.int32)
    dtypes = [tf.int32, tf.float64]
    capacity = 10
    memory_limit = 1024
    container = "test_container"
    shared_name = "test_shared_name"
    name = "test_op"

    input_dict = {
        "key": key,
        "indices": indices,
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    key = np.array(3, dtype=np.int64)
    indices = np.array([], dtype=np.int32)
    dtypes = [tf.string]
    capacity = 5
    memory_limit = 512
    container = "another_container"
    shared_name = "another_shared_name"
    name = None

    input_dict = {
        "key": key,
        "indices": indices,
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    key = np.array(4, dtype=np.int64)
    indices = np.array([0, 1, 2], dtype=np.int32)
    dtypes = [tf.bool, tf.complex64, tf.uint8]
    capacity = 1
    memory_limit = 64
    container = ""
    shared_name = ""
    name = "op_name"

    input_dict = {
        "key": key,
        "indices": indices,
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    key = np.array(-5, dtype=np.int64)
    indices = np.array([5], dtype=np.int32)
    dtypes = [tf.int64]
    capacity = 0
    memory_limit = 0
    container = ""
    shared_name = ""
    name = None

    input_dict = {
        "key": key,
        "indices": indices,
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    key = np.array(6, dtype=np.int64)
    indices = np.array([0, 1], dtype=np.int32)
    dtypes = [tf.bfloat16, tf.float16]
    capacity = 20
    memory_limit = 2048
    container = "another_test_container"
    shared_name = "another_test_shared_name"
    name = "test_op2"

    input_dict = {
        "key": key,
        "indices": indices,
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    key = np.array(7, dtype=np.int64)
    indices = np.array([2,3,4,5], dtype=np.int32)
    dtypes = [tf.qint8, tf.quint8, tf.qint16, tf.quint16]
    capacity = 1
    memory_limit = 16
    container = ""
    shared_name = ""
    name = None

    input_dict = {
        "key": key,
        "indices": indices,
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    key = np.array(8, dtype=np.int64)
    indices = np.array([0], dtype=np.int32)
    dtypes = [tf.resource]
    capacity = 3
    memory_limit = 256
    container = "resource_container"
    shared_name = "resource_shared_name"
    name = "resource_op"

    input_dict = {
        "key": key,
        "indices": indices,
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    key = np.array(9, dtype=np.int64)
    indices = np.array([0,1,2,3,4,5,6], dtype=np.int32)
    dtypes = [tf.variant, tf.complex128, tf.uint32, tf.int8, tf.uint16, tf.uint64, tf.int16]
    capacity = 1
    memory_limit = 32
    container = ""
    shared_name = ""
    name = "variant_op"

    input_dict = {
        "key": key,
        "indices": indices,
        "dtypes": dtypes,
        "capacity": capacity,
        "memory_limit": memory_limit,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10
    key = np.array(10, dtype=np.int64)
    indices = np.array([0,1,2,3], dtype=np.int32)
    dtypes = [tf.float32, tf.float64, tf.int32, tf.int64]
    capacity = 5
    memory_limit = 128
    container = "new_container"
    shared_name = "new_shared_name"
    name = None

    input_dict = {
        "key": key,
        "indices": indices,
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
generated_inputs["tf.raw_ops.OrderedMapUnstage"] = tf_raw_ops_ordered_map_unstage_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.OrderedMapUnstage' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.OrderedMapUnstage'.")

check_valid('tf.raw_ops.OrderedMapUnstage', generated_inputs['tf.raw_ops.OrderedMapUnstage'], lib="tf", suffix=0)
