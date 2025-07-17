
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_deserialize_many_sparse_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    serialized_sparse = np.array([
        [b'\x08\x01\x12\x06\x08\x00\x10\x00\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x01\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x02\x1a\x02\x08\x01'],
        [b'\x08\x01\x12\x06\x08\x00\x10\x03\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x04\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x05\x1a\x02\x08\x01']
    ], dtype=np.string_)
    dtype = np.int64
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "name": "test"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different dtype (float32)
    serialized_sparse = np.array([
        [b'\x08\x01\x12\x06\x08\x00\x10\x00\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x01\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x02\x1a\x02\x08\x01'],
        [b'\x08\x01\x12\x06\x08\x00\x10\x03\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x04\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x05\x1a\x02\x08\x01']
    ], dtype=np.string_)
    dtype = np.float32
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "name": "test2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty serialized data
    serialized_sparse = np.array([
        [b'', b'', b''],
        [b'', b'', b'']
    ], dtype=np.string_)
    dtype = np.int32
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "name": "test3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger minibatch size
    serialized_sparse = np.array([
        [b'\x08\x01\x12\x06\x08\x00\x10\x00\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x01\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x02\x1a\x02\x08\x01'],
        [b'\x08\x01\x12\x06\x08\x00\x10\x03\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x04\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x05\x1a\x02\x08\x01'],
        [b'\x08\x01\x12\x06\x08\x00\x10\x06\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x07\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x08\x1a\x02\x08\x01']
    ], dtype=np.string_)
    dtype = np.int64
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "name": "test4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Uint8
    serialized_sparse = np.array([
        [b'\x08\x01\x12\x06\x08\x00\x10\x00\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x01\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x02\x1a\x02\x08\x01'],
        [b'\x08\x01\x12\x06\x08\x00\x10\x03\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x04\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x05\x1a\x02\x08\x01']
    ], dtype=np.string_)
    dtype = np.uint8
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "name": "test5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Int16
    serialized_sparse = np.array([
        [b'\x08\x01\x12\x06\x08\x00\x10\x00\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x01\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x02\x1a\x02\x08\x01'],
        [b'\x08\x01\x12\x06\x08\x00\x10\x03\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x04\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x05\x1a\x02\x08\x01']
    ], dtype=np.string_)
    dtype = np.int16
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "name": "test6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Int32
    serialized_sparse = np.array([
        [b'\x08\x01\x12\x06\x08\x00\x10\x00\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x01\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x02\x1a\x02\x08\x01'],
        [b'\x08\x01\x12\x06\x08\x00\x10\x03\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x04\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x05\x1a\x02\x08\x01']
    ], dtype=np.string_)
    dtype = np.int32
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "name": "test7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float16
    serialized_sparse = np.array([
        [b'\x08\x01\x12\x06\x08\x00\x10\x00\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x01\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x02\x1a\x02\x08\x01'],
        [b'\x08\x01\x12\x06\x08\x00\x10\x03\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x04\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x05\x1a\x02\x08\x01']
    ], dtype=np.string_)
    dtype = np.float16
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "name": "test8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different minibatch and smaller rank
    serialized_sparse = np.array([
        [b'\x08\x01\x12\x06\x08\x00\x10\x00\x1a\x02\x08\x01'],
        [b'\x08\x01\x12\x06\x08\x00\x10\x03\x1a\x02\x08\x01'],
        [b'\x08\x01\x12\x06\x08\x00\x10\x06\x1a\x02\x08\x01']
    ], dtype=np.string_)
    dtype = np.int64
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "name": "test9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Boolean
    serialized_sparse = np.array([
        [b'\x08\x01\x12\x06\x08\x00\x10\x00\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x01\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x02\x1a\x02\x08\x01'],
        [b'\x08\x01\x12\x06\x08\x00\x10\x03\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x04\x1a\x02\x08\x01',
         b'\x08\x01\x12\x06\x08\x00\x10\x05\x1a\x02\x08\x01']
    ], dtype=np.string_)
    dtype = np.bool_
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "name": "test10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.DeserializeManySparse"] = tf_raw_ops_deserialize_many_sparse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DeserializeManySparse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DeserializeManySparse'.")

check_valid('tf.raw_ops.DeserializeManySparse', generated_inputs['tf.raw_ops.DeserializeManySparse'], lib="tf", suffix=0)
