
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def check_valid(api, input_list, lib="tf", suffix="0"):
    for i, input_dict in enumerate(input_list):
        try:
            if lib == "tf":
                api = eval(api)
            inp = {"args": [], "kwargs": input_dict}
            output = run_api(api, input_dict, cpu=True, lib=lib)
        except Exception as e:
            print(f"Error in input {i}: {e}")

def run_api(func, input_dict, cpu=True, lib="tf"):
    if lib == "torch":
        inp = {"args": [], "kwargs": input_dict}
        result = func(*inp["args"], **inp["kwargs"])
    elif lib == "tf":
        result = func(**input_dict)
    return result

def tf_raw_ops_deserialize_many_sparse_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    serialized_sparse = np.array([
        [b'\x08\x01*\x00\x12\x08\x08\x01\x10\x00\x18\x01"\x08\x08\x01\x10\x00\x18\x00', b'\x08\x01*\x00\x12\x08\x08\x01\x10\x00\x18\x01"\x08\x08\x01\x10\x00\x18\x00', b'\x08\x01*\x00\x12\x08\x08\x01\x10\x00\x18\x01"\x08\x08\x01\x10\x00\x18\x00'],
        [b'\x08\x01*\x00\x12\x08\x08\x01\x10\x00\x18\x01"\x08\x08\x01\x10\x00\x18\x00', b'\x08\x01*\x00\x12\x08\x08\x01\x10\x00\x18\x01"\x08\x08\x01\x10\x00\x18\x00', b'\x08\x01*\x00\x12\x08\x08\x01\x10\x00\x18\x01"\x08\x08\x01\x10\x00\x18\x00']
    ], dtype=np.object_)
    dtype = tf.float32
    name = "sparse_deserialization_1"

    input_dict = {
        "serialized_sparse": serialized_sparse,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different dtype
    serialized_sparse = np.array([
        [b'\x08\x01*\x00\x12\x08\x08\x01\x10\x00\x18\x01"\x08\x08\x01\x10\x00\x18\x00', b'\x08\x01*\x00\x12\x08\x08\x01\x10\x00\x18\x01"\x08\x08\x01\x10\x00\x18\x00', b'\x08\x01*\x00\x12\x08\x08\x01\x10\x00\x18\x01"\x08\x08\x01\x10\x00\x18\x00'],
        [b'\x08\x01*\x00\x12\x08\x08\x01\x10\x00\x18\x01"\x08\x08\x01\x10\x00\x18\x00', b'\x08\x01*\x00\x12\x08\x08\x01\x10\x00\x18\x01"\x08\x08\x01\x10\x00\x18\x00', b'\x08\x01*\x00\x12\x08\x08\x01\x10\x00\x18\x01"\x08\x08\x01\x10\x00\x18\x00']
    ], dtype=np.object_)
    dtype = tf.int32
    name = "sparse_deserialization_2"

    input_dict = {
        "serialized_sparse": serialized_sparse,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 3: Different name
    serialized_sparse = np.array([
        [b'\x08\x01*\x00\x12\x08\x08\x01\x10\x00\x18\x01"\x08\x08\x01\x10\x00\x18\x00', b'\x08\x01*\x00\x12\x08\x08\x01\x10\x00\x18\x01"\x08\x08\x01\x10\x00\x18\x00', b'\x08\x01*\x00\x12\x08\x08\x01\x10\x00\x18\x01"\x08\x08\x01\x10\x00\x18\x00'],
        [b'\x08\x01*\x00\x12\x08\x08\x01\x10\x00\x18\x01"\x08\x08\x01\x10\x00\x18\x00', b'\x08\x01*\x00\x12\x08\x08\x01\x10\x00\x18\x01"\x08\x08\x01\x10\x00\x18\x00', b'\x08\x01*\x00\x12\x08\x08\x01\x10\x00\x18\x01"\x08\x08\x01\x10\x00\x18\x00']
    ], dtype=np.object_)
    dtype = tf.float32
    name = "another_sparse_name"

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
