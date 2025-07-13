
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_deserialize_many_sparse_inputs():
    list_of_inputs = []

    # Input 1
    serialized_sparse = np.array([
        [b'\x08\x00\x10\x02\x1a\x04\x08\x01\x10\x01', b'\x12\x04\x00\x01\x02\x03', b'\x1a\x08\x08\x02\x10\x02\x18\x04']
    ], dtype=np.object_)
    dtype = tf.float32
    name = "sparse_tensor_1"

    input_dict = {
        "serialized_sparse": serialized_sparse,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    serialized_sparse = np.array([
        [b'\x08\x00\x10\x02\x1a\x04\x08\x01\x10\x01', b'\x12\x04\x00\x01\x02\x03', b'\x1a\x08\x08\x02\x10\x02\x18\x04'],
        [b'\x08\x00\x10\x02\x1a\x04\x08\x01\x10\x01', b'\x12\x04\x04\x05\x06\x07', b'\x1a\x08\x08\x02\x10\x02\x18\x04']
    ], dtype=np.object_)
    dtype = tf.int32
    name = "sparse_tensor_2"

    input_dict = {
        "serialized_sparse": serialized_sparse,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 3
    serialized_sparse = np.array([
        [b'\x08\x00\x10\x02\x1a\x04\x08\x01\x10\x01', b'\x12\x04\x00\x01\x02\x03', b'\x1a\x08\x08\x02\x10\x02\x18\x04']
    ], dtype=np.object_)
    dtype = tf.float64
    name = "sparse_tensor_3"

    input_dict = {
        "serialized_sparse": serialized_sparse,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    serialized_sparse = np.array([
        [b'\x08\x00\x10\x02\x1a\x04\x08\x01\x10\x01', b'\x12\x04\x00\x01\x02\x03', b'\x1a\x08\x08\x02\x10\x02\x18\x04'],
        [b'\x08\x00\x10\x02\x1a\x04\x08\x01\x10\x01', b'\x12\x04\x04\x05\x06\x07', b'\x1a\x08\x08\x02\x10\x02\x18\x04'],
        [b'\x08\x00\x10\x02\x1a\x04\x08\x01\x10\x01', b'\x12\x04\x08\t\n\x0b', b'\x1a\x08\x08\x02\x10\x02\x18\x04']
    ], dtype=np.object_)
    dtype = tf.int64
    name = "sparse_tensor_4"

    input_dict = {
        "serialized_sparse": serialized_sparse,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    serialized_sparse = np.array([
        [b'\x08\x00\x10\x02\x1a\x04\x08\x01\x10\x01', b'\x12\x04\x00\x01\x02\x03', b'\x1a\x08\x08\x02\x10\x02\x18\x04']
    ], dtype=np.object_)
    dtype = tf.uint8
    name = "sparse_tensor_5"

    input_dict = {
        "serialized_sparse": serialized_sparse,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    serialized_sparse = np.array([
        [b'\x08\x00\x10\x02\x1a\x04\x08\x01\x10\x01', b'\x12\x04\x00\x01\x02\x03', b'\x1a\x08\x08\x02\x10\x02\x18\x04'],
        [b'\x08\x00\x10\x02\x1a\x04\x08\x01\x10\x01', b'\x12\x04\x04\x05\x06\x07', b'\x1a\x08\x08\x02\x10\x02\x18\x04']
    ], dtype=np.object_)
    dtype = tf.uint16
    name = "sparse_tensor_6"

    input_dict = {
        "serialized_sparse": serialized_sparse,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    serialized_sparse = np.array([
        [b'\x08\x00\x10\x02\x1a\x04\x08\x01\x10\x01', b'\x12\x04\x00\x01\x02\x03', b'\x1a\x08\x08\x02\x10\x02\x18\x04']
    ], dtype=np.object_)
    dtype = tf.uint32
    name = "sparse_tensor_7"

    input_dict = {
        "serialized_sparse": serialized_sparse,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    serialized_sparse = np.array([
        [b'\x08\x00\x10\x02\x1a\x04\x08\x01\x10\x01', b'\x12\x04\x00\x01\x02\x03', b'\x1a\x08\x08\x02\x10\x02\x18\x04'],
        [b'\x08\x00\x10\x02\x1a\x04\x08\x01\x10\x01', b'\x12\x04\x04\x05\x06\x07', b'\x1a\x08\x08\x02\x10\x02\x18\x04']
    ], dtype=np.object_)
    dtype = tf.uint64
    name = "sparse_tensor_8"

    input_dict = {
        "serialized_sparse": serialized_sparse,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    serialized_sparse = np.array([
        [b'\x08\x00\x10\x02\x1a\x04\x08\x01\x10\x01', b'\x12\x04\x00\x01\x02\x03', b'\x1a\x08\x08\x02\x10\x02\x18\x04']
    ], dtype=np.object_)
    dtype = tf.int8
    name = "sparse_tensor_9"

    input_dict = {
        "serialized_sparse": serialized_sparse,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10
    serialized_sparse = np.array([
        [b'\x08\x00\x10\x02\x1a\x04\x08\x01\x10\x01', b'\x12\x04\x00\x01\x02\x03', b'\x1a\x08\x08\x02\x10\x02\x18\x04'],
        [b'\x08\x00\x10\x02\x1a\x04\x08\x01\x10\x01', b'\x12\x04\x04\x05\x06\x07', b'\x1a\x08\x08\x02\x10\x02\x18\x04']
    ], dtype=np.object_)
    dtype = tf.bfloat16
    name = "sparse_tensor_10"

    input_dict = {
        "serialized_sparse": serialized_sparse,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DeserializeManySparse"] = tf_raw_ops_deserialize_many_sparse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DeserializeManySparse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DeserializeManySparse'.")

check_valid('tf.raw_ops.DeserializeManySparse', generated_inputs['tf.raw_ops.DeserializeManySparse'], lib="tf", suffix=0)
