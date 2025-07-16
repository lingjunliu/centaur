
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_deserialize_many_sparse_inputs():
    list_of_inputs = []

    # Input 1
    serialized_sparse = np.array([
        [b'\x08\x00\x00\x00', b'\x08\x00\x00\x00', b'\x08\x00\x00\x00']
    ], dtype=np.string_)
    dtype = np.int32
    rank = 1
    name = "sparse_tensor_1"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    serialized_sparse = np.array([
        [b'\x08\x00\x00\x00', b'\x08\x00\x00\x00', b'\x08\x00\x00\x00']
    ], dtype=np.string_)
    dtype = np.int32
    rank = 1
    name = "sparse_tensor_2"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    serialized_sparse = np.array([
        [b'\x08\x00\x00\x00', b'\x08\x00\x00\x00', b'\x08\x00\x00\x00']
    ], dtype=np.string_)
    dtype = np.int32
    rank = 1
    name = "sparse_tensor_3"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4
    serialized_sparse = np.array([
        [b'\x08\x00\x00\x00', b'\x08\x00\x00\x00', b'\x08\x00\x00\x00']
    ], dtype=np.string_)
    dtype = np.int32
    rank = 1
    name = None
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    serialized_sparse = np.array([
        [b'\x08\x00\x00\x00', b'\x08\x00\x00\x00', b'\x08\x00\x00\x00']
    ], dtype=np.string_)
    dtype = np.int32
    rank = None
    name = "sparse_tensor_5"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    serialized_sparse = np.array([
        [b'\x04\x00\x00\x00', b'\x04\x00\x00\x00', b'\x04\x00\x00\x00']
    ], dtype=np.string_)
    dtype = np.int32
    rank = 1
    name = "sparse_tensor_6"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    serialized_sparse = np.array([
        [b'\x08\x00\x00\x00', b'\x08\x00\x00\x00', b'\x08\x00\x00\x00']
    ], dtype=np.string_)
    dtype = np.int32
    rank = 1
    name = "sparse_tensor_7"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    serialized_sparse = np.array([
        [b'\x08\x00\x00\x00', b'\x08\x00\x00\x00', b'\x08\x00\x00\x00']
    ], dtype=np.string_)
    dtype = np.int32
    rank = 1
    name = "sparse_tensor_8"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    serialized_sparse = np.array([
        [b'\x02\x00', b'\x02\x00', b'\x02\x00']
    ], dtype=np.string_)
    dtype = np.int16
    rank = 1
    name = "sparse_tensor_9"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    serialized_sparse = np.array([
        [b'\x01', b'\x01', b'\x01']
    ], dtype=np.string_)
    dtype = np.int8
    rank = 1
    name = "sparse_tensor_10"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.deserialize_many_sparse"] = tf_io_deserialize_many_sparse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.deserialize_many_sparse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.deserialize_many_sparse'.")

check_valid('tf.io.deserialize_many_sparse', generated_inputs['tf.io.deserialize_many_sparse'], lib="tf", suffix=0)
