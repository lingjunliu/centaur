
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_lookup_StaticHashTable_inputs():
    list_of_inputs = []

    # Input 1: Basic test with integers
    keys_tensor = tf.constant([0, 1, 2])
    vals_tensor = tf.constant([1, 2, 3])
    default_value = tf.constant(-1)
    name = "table1"
    input_dict = {
        "initializer": keys_tensor,
        "default_value": default_value,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different default value
    keys_tensor = tf.constant([0, 1, 2])
    vals_tensor = tf.constant([1, 2, 3])
    default_value = tf.constant(100)
    name = "table2"
    input_dict = {
        "initializer": keys_tensor,
        "default_value": default_value,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty initializer
    keys_tensor = tf.constant([])
    vals_tensor = tf.constant([])
    default_value = tf.constant(-1)
    name = "table3"
    input_dict = {
        "initializer": keys_tensor,
        "default_value": default_value,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: More keys and values
    keys_tensor = tf.constant([0, 1, 2, 3, 4])
    vals_tensor = tf.constant([1, 2, 3, 4, 5])
    default_value = tf.constant(-1)
    name = "table4"
    input_dict = {
        "initializer": keys_tensor,
        "default_value": default_value,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different data types (float)
    keys_tensor = tf.constant([1.0, 2.0, 3.0])
    vals_tensor = tf.constant([4.0, 5.0, 6.0])
    default_value = tf.constant(0.0)
    name = "table5"
    input_dict = {
        "initializer": keys_tensor,
        "default_value": default_value,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative values in keys and vals
    keys_tensor = tf.constant([-1, -2, -3])
    vals_tensor = tf.constant([-4, -5, -6])
    default_value = tf.constant(0)
    name = "table6"
    input_dict = {
        "initializer": keys_tensor,
        "default_value": default_value,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D keys and values
    keys_tensor = tf.constant([[1, 2], [3, 4]])
    vals_tensor = tf.constant([[5, 6], [7, 8]])
    default_value = tf.constant(-1)
    name = "table7"
    input_dict = {
        "initializer": keys_tensor,
        "default_value": default_value,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different default value with the same dtype as values
    keys_tensor = tf.constant([0, 1, 2])
    vals_tensor = tf.constant([1, 2, 3], dtype=tf.int64)
    default_value = tf.constant(-1, dtype=tf.int64)
    name = "table8"
    input_dict = {
        "initializer": keys_tensor,
        "default_value": default_value,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty table
    keys_tensor = tf.constant([], dtype=tf.int32)
    vals_tensor = tf.constant([], dtype=tf.int32)
    default_value = tf.constant(-1, dtype=tf.int32)
    name = "table9"
    input_dict = {
        "initializer": keys_tensor,
        "default_value": default_value,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32 table
    keys_tensor = tf.constant([1.1, 2.2, 3.3], dtype=tf.float32)
    vals_tensor = tf.constant([4.4, 5.5, 6.6], dtype=tf.float32)
    default_value = tf.constant(0.0, dtype=tf.float32)
    name = "table10"
    input_dict = {
        "initializer": keys_tensor,
        "default_value": default_value,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.lookup.StaticHashTable"] = tf_lookup_StaticHashTable_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.lookup.StaticHashTable' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.lookup.StaticHashTable'.")

check_valid('tf.lookup.StaticHashTable', generated_inputs['tf.lookup.StaticHashTable'], lib="tf", suffix=0)
