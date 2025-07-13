
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_stagepeek_inputs():
    list_of_inputs = []

    # Input 1
    index = np.array(0, dtype=np.int32)
    dtypes = [tf.float32]
    capacity = 10
    memory_limit = 1024
    container = "test_container_1"
    shared_name = "test_shared_name_1"
    name = "test_name_1"
    input_dict = {"index": index, "dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    index = np.array(1, dtype=np.int32)
    dtypes = [tf.int32, tf.float64]
    capacity = 5
    memory_limit = 512
    container = "test_container_2"
    shared_name = "test_shared_name_2"
    name = "test_name_2"
    input_dict = {"index": index, "dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    index = np.array(2, dtype=np.int32)
    dtypes = [tf.string]
    capacity = 1
    memory_limit = 0
    container = ""
    shared_name = ""
    name = None
    input_dict = {"index": index, "dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    index = np.array(0, dtype=np.int32)
    dtypes = [tf.bool, tf.int8, tf.int16]
    capacity = 0
    memory_limit = 0
    container = "container_4"
    shared_name = "shared_4"
    name = "name_4"
    input_dict = {"index": index, "dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    index = np.array(5, dtype=np.int32)
    dtypes = [tf.uint8, tf.uint16, tf.uint32, tf.uint64]
    capacity = 15
    memory_limit = 2048
    container = "container_5"
    shared_name = "shared_5"
    name = "name_5"
    input_dict = {"index": index, "dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    index = np.array(3, dtype=np.int32)
    dtypes = [tf.float16, tf.float32, tf.float64]
    capacity = 3
    memory_limit = 768
    container = "container_6"
    shared_name = "shared_6"
    name = "name_6"
    input_dict = {"index": index, "dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    index = np.array(0, dtype=np.int32)
    dtypes = [tf.complex64, tf.complex128]
    capacity = 7
    memory_limit = 128
    container = "container_7"
    shared_name = "shared_7"
    name = "name_7"
    input_dict = {"index": index, "dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    index = np.array(10, dtype=np.int32)
    dtypes = [tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32]
    capacity = 20
    memory_limit = 4096
    container = "container_8"
    shared_name = "shared_8"
    name = "name_8"
    input_dict = {"index": index, "dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    index = np.array(0, dtype=np.int32)
    dtypes = [tf.resource]
    capacity = 4
    memory_limit = 384
    container = "container_9"
    shared_name = "shared_9"
    name = "name_9"
    input_dict = {"index": index, "dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    index = np.array(1, dtype=np.int32)
    dtypes = [tf.variant]
    capacity = 8
    memory_limit = 1536
    container = "container_10"
    shared_name = "shared_10"
    name = "name_10"
    input_dict = {"index": index, "dtypes": dtypes, "capacity": capacity, "memory_limit": memory_limit, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.StagePeek"] = tf_raw_ops_stagepeek_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.StagePeek' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StagePeek'.")

check_valid('tf.raw_ops.StagePeek', generated_inputs['tf.raw_ops.StagePeek'], lib="tf", suffix=0)
