
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MapUnstage_inputs():
    list_of_inputs = []

    # Input 1
    key = np.array(1, dtype=np.int64)
    indices = np.array([0], dtype=np.int32)
    dtypes = [tf.float32]
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

    # Input 2
    key = np.array(2, dtype=np.int64)
    indices = np.array([1, 2, 3], dtype=np.int32)
    dtypes = [tf.int32, tf.float64]
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

    # Input 3
    key = np.array(3, dtype=np.int64)
    indices = np.array([], dtype=np.int32)
    dtypes = [tf.string]
    capacity = 5
    memory_limit = 512
    container = "another_container"
    shared_name = "another_shared_name"
    name = "another_op"

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
    key = np.array(-1, dtype=np.int64)
    indices = np.array([0, 1], dtype=np.int32)
    dtypes = [tf.bool, tf.int64, tf.uint8]
    capacity = 1
    memory_limit = 128
    container = "neg_container"
    shared_name = "neg_shared_name"
    name = "neg_op"

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
    key = np.array(0, dtype=np.int64)
    indices = np.array([2, 3, 4, 5], dtype=np.int32)
    dtypes = [tf.complex64]
    capacity = 100
    memory_limit = 10240
    container = "zero_container"
    shared_name = "zero_shared_name"
    name = "zero_op"

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
    key = np.array(2**31, dtype=np.int64)
    indices = np.array([2], dtype=np.int32)
    dtypes = [tf.int16]
    capacity = 200
    memory_limit = 20480
    container = "big_key_container"
    shared_name = "big_key_shared_name"
    name = "big_key_op"

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
    key = np.array(-(2**31), dtype=np.int64)
    indices = np.array([2,3], dtype=np.int32)
    dtypes = [tf.float16, tf.bfloat16]
    capacity = 300
    memory_limit = 30480
    container = "negative_key_container"
    shared_name = "negative_key_shared_name"
    name = "negative_key_op"

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
    key = np.array(5, dtype=np.int64)
    indices = np.array([0,1,2,3,4], dtype=np.int32)
    dtypes = [tf.qint8]
    capacity = 400
    memory_limit = 40480
    container = "qint8_container"
    shared_name = "qint8_shared_name"
    name = "qint8_op"

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
    key = np.array(6, dtype=np.int64)
    indices = np.array([0,1], dtype=np.int32)
    dtypes = [tf.quint8]
    capacity = 500
    memory_limit = 50480
    container = "quint8_container"
    shared_name = "quint8_shared_name"
    name = "quint8_op"

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
    key = np.array(7, dtype=np.int64)
    indices = np.array([0,1,2], dtype=np.int32)
    dtypes = [tf.qint32]
    capacity = 600
    memory_limit = 60480
    container = "qint32_container"
    shared_name = "qint32_shared_name"
    name = "qint32_op"

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
generated_inputs["tf.raw_ops.MapUnstage"] = tf_raw_ops_MapUnstage_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MapUnstage' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MapUnstage'.")

check_valid('tf.raw_ops.MapUnstage', generated_inputs['tf.raw_ops.MapUnstage'], lib="tf", suffix=0)
