
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_lookup_static_hash_table_inputs():
    list_of_inputs = []

    # Input 1: Simple case with string keys and integer values
    keys_tensor = tf.constant(['a', 'b', 'c'])
    vals_tensor = tf.constant([1, 2, 3])
    default_value = tf.constant(-1, dtype=tf.int32)
    name = "table1"
    input_dict = {"initializer": (keys_tensor, vals_tensor), "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer keys and float values
    keys_tensor = tf.constant([1, 2, 3])
    vals_tensor = tf.constant([1.1, 2.2, 3.3])
    default_value = tf.constant(0.0, dtype=tf.float32)
    name = "table2"
    input_dict = {"initializer": (keys_tensor, vals_tensor), "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float keys and string values
    keys_tensor = tf.constant([1.1, 2.2, 3.3])
    vals_tensor = tf.constant(['x', 'y', 'z'])
    default_value = tf.constant("unknown", dtype=tf.string)
    name = "table3"
    input_dict = {"initializer": (keys_tensor, vals_tensor), "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty keys and values
    keys_tensor = tf.constant([], dtype=tf.string)
    vals_tensor = tf.constant([], dtype=tf.int32)
    default_value = tf.constant(-1, dtype=tf.int32)
    name = "table4"
    input_dict = {"initializer": (keys_tensor, vals_tensor), "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative integer keys and values
    keys_tensor = tf.constant([-1, -2, -3])
    vals_tensor = tf.constant([-4, -5, -6])
    default_value = tf.constant(0, dtype=tf.int32)
    name = "table5"
    input_dict = {"initializer": (keys_tensor, vals_tensor), "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different shapes for keys and values (should still work if the initializer handles it)
    keys_tensor = tf.constant([['a', 'b'], ['c', 'd']])
    vals_tensor = tf.constant([[1, 2], [3, 4]])
    default_value = tf.constant(-1, dtype=tf.int32)
    name = "table6"
    input_dict = {"initializer": (keys_tensor, vals_tensor), "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Keys and values with higher dimension
    keys_tensor = tf.constant([[['a', 'b'], ['c', 'd']]])
    vals_tensor = tf.constant([[[1, 2], [3, 4]]])
    default_value = tf.constant(-1, dtype=tf.int32)
    name = "table7"
    input_dict = {"initializer": (keys_tensor, vals_tensor), "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Bool type values
    keys_tensor = tf.constant(['a', 'b', 'c'])
    vals_tensor = tf.constant([True, False, True])
    default_value = tf.constant(False, dtype=tf.bool)
    name = "table8"
    input_dict = {"initializer": (keys_tensor, vals_tensor), "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Unicode characters
    keys_tensor = tf.constant(['你好', '世界'])
    vals_tensor = tf.constant([1, 2])
    default_value = tf.constant(-1, dtype=tf.int32)
    name = "table9"
    input_dict = {"initializer": (keys_tensor, vals_tensor), "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different default value
    keys_tensor = tf.constant(['a', 'b', 'c'])
    vals_tensor = tf.constant([1, 2, 3])
    default_value = tf.constant(100, dtype=tf.int32)
    name = "table10"
    input_dict = {"initializer": (keys_tensor, vals_tensor), "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.lookup.StaticHashTable"] = tf_lookup_static_hash_table_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.lookup.StaticHashTable' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.lookup.StaticHashTable'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.lookup.StaticHashTable', generated_inputs['tf.lookup.StaticHashTable'], lib="tf", suffix=0)
