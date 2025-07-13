
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MapPeek_inputs():
    list_of_inputs = []

    # Input 1
    key = np.array(1, dtype=np.int64)
    indices = np.array([0], dtype=np.int32)
    dtypes = [tf.float32]
    capacity = 10
    memory_limit = 1024
    container = "test_container"
    shared_name = "test_shared_name"
    name = "test_name"

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
    dtypes = [tf.int32]
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
    indices = np.array([[0, 1], [2, 3]], dtype=np.int32)
    dtypes = [tf.string]
    capacity = 5
    memory_limit = 512
    container = "another_container"
    shared_name = "another_shared_name"
    name = "another_name"

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
    indices = np.array([0, -1, 2], dtype=np.int32)
    dtypes = [tf.bool]
    capacity = 1
    memory_limit = 128
    container = "neg_container"
    shared_name = "neg_shared_name"
    name = "neg_name"

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
    indices = np.array([[-1, 0], [1, -1]], dtype=np.int32)
    dtypes = [tf.float64]
    capacity = 20
    memory_limit = 2048
    container = ""
    shared_name = "zero_shared"
    name = "zero_name"

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
    key = np.array(1000, dtype=np.int64)
    indices = np.array([10, 20, 30], dtype=np.int32)
    dtypes = [tf.complex64]
    capacity = 100
    memory_limit = 10240
    container = "thousand_container"
    shared_name = "thousand_shared"
    name = "thousand_name"

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
    key = np.array(-100, dtype=np.int64)
    indices = np.array([[-1, -2], [-3, -4]], dtype=np.int32)
    dtypes = [tf.uint8]
    capacity = 50
    memory_limit = 5120
    container = "negative_container"
    shared_name = "negative_shared"
    name = "negative_name"

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
    key = np.array(2**30, dtype=np.int64)
    indices = np.array([0, 1], dtype=np.int32)
    dtypes = [tf.qint8]
    capacity = 50
    memory_limit = 5120
    container = "bigint_container"
    shared_name = "bigint_shared"
    name = "bigint_name"

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
    key = np.array(-(2**30), dtype=np.int64)
    indices = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    dtypes = [tf.resource]
    capacity = 50
    memory_limit = 5120
    container = "negative_bigint_container"
    shared_name = "negative_bigint_shared"
    name = "negative_bigint_name"

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
    key = np.array(0, dtype=np.int64)
    indices = np.array([], dtype=np.int32)
    dtypes = [tf.float32]
    capacity = 10
    memory_limit = 1024
    container = "test_container"
    shared_name = "test_shared_name"
    name = "test_name"

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
generated_inputs["tf.raw_ops.MapPeek"] = tf_raw_ops_MapPeek_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MapPeek' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MapPeek'.")

check_valid('tf.raw_ops.MapPeek', generated_inputs['tf.raw_ops.MapPeek'], lib="tf", suffix=0)
