
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_lookup_static_hash_table_inputs():
    list_of_inputs = []

    # Input 1
    keys_tensor = tf.constant(['a', 'b', 'c'])
    vals_tensor = tf.constant([7, 8, 9], dtype=tf.int32)
    default_value = tf.constant(-1, dtype=tf.int32)
    name = "table1"
    input_dict = {"initializer": keys_tensor, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    keys_tensor = tf.constant([1, 2, 3], dtype=tf.int32)
    vals_tensor = tf.constant([1, 2, 3], dtype=tf.int32)
    default_value = tf.constant(0, dtype=tf.int32)
    name = "table2"
    input_dict = {"initializer": keys_tensor, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    keys_tensor = tf.constant([1, 2, 3], dtype=tf.int64)
    vals_tensor = tf.constant([1, 2, 3], dtype=tf.int64)
    default_value = tf.constant(-1, dtype=tf.int64)
    name = "table3"
    input_dict = {"initializer": keys_tensor, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    keys_tensor = tf.constant(['a', 'b', 'c', 'a'])
    vals_tensor = tf.constant([7, 8, 9, 10], dtype=tf.int32)
    default_value = tf.constant(-1, dtype=tf.int32)
    name = "table4"
    input_dict = {"initializer": keys_tensor, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    keys_tensor = tf.constant([['a', 'b'], ['c', 'd']])
    vals_tensor = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)
    default_value = tf.constant(-1, dtype=tf.int32)
    name = "table5"
    input_dict = {"initializer": keys_tensor, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    keys_tensor = tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)
    vals_tensor = tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)
    default_value = tf.constant(0, dtype=tf.int32)
    name = "table6"
    input_dict = {"initializer": keys_tensor, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    keys_tensor = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    vals_tensor = tf.constant([1, 2, 3], dtype=tf.int32)
    default_value = tf.constant(-1, dtype=tf.int32)
    name = "table7"
    input_dict = {"initializer": keys_tensor, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    keys_tensor = tf.constant(['a'])
    vals_tensor = tf.constant([7], dtype=tf.int32)
    default_value = tf.constant(-1, dtype=tf.int32)
    name = "table8"
    input_dict = {"initializer": keys_tensor, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    keys_tensor = tf.constant([1, 2, 3], dtype=tf.int32)
    vals_tensor = tf.constant([1, 2, 3], dtype=tf.int32)
    default_value = tf.constant(0, dtype=tf.int32)
    name = "table9"
    input_dict = {"initializer": keys_tensor, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    keys_tensor = tf.constant(['a', 'b', 'c'])
    vals_tensor = tf.constant([7, 8, 9], dtype=tf.int32)
    default_value = tf.constant(-1, dtype=tf.int32)
    name = "table10"
    input_dict = {"initializer": keys_tensor, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.lookup.StaticHashTable"] = tf_lookup_static_hash_table_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.lookup.StaticHashTable' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.lookup.StaticHashTable'.")

check_valid('tf.lookup.StaticHashTable', generated_inputs['tf.lookup.StaticHashTable'], lib="tf", suffix=0)
