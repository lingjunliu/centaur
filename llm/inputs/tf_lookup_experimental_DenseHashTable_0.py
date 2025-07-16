
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_lookup_experimental_dense_hash_table_inputs():
    list_of_inputs = []

    # Input 1
    key_dtype = tf.string
    value_dtype = tf.int64
    default_value = tf.constant(-1, dtype=value_dtype)
    empty_key = tf.constant("", dtype=key_dtype)
    deleted_key = tf.constant("$", dtype=key_dtype)
    name = "table1"
    input_dict = {
        "key_dtype": key_dtype,
        "value_dtype": value_dtype,
        "default_value": default_value,
        "empty_key": empty_key,
        "deleted_key": deleted_key,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    key_dtype = tf.int32
    value_dtype = tf.float32
    default_value = tf.constant(0.0, dtype=value_dtype)
    empty_key = tf.constant(-1, dtype=key_dtype)
    deleted_key = tf.constant(-2, dtype=key_dtype)
    name = "table2"
    input_dict = {
        "key_dtype": key_dtype,
        "value_dtype": value_dtype,
        "default_value": default_value,
        "empty_key": empty_key,
        "deleted_key": deleted_key,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.lookup.experimental.DenseHashTable"] = tf_lookup_experimental_dense_hash_table_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.lookup.experimental.DenseHashTable' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.lookup.experimental.DenseHashTable'.")

check_valid('tf.lookup.experimental.DenseHashTable', generated_inputs['tf.lookup.experimental.DenseHashTable'], lib="tf", suffix=0)
