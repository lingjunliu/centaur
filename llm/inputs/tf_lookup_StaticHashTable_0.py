
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_lookup_static_hash_table_inputs():
    list_of_inputs = []

    # Input 1: Basic string keys and int values
    keys_tensor = tf.constant(np.array(['a', 'b', 'c']))
    vals_tensor = tf.constant(np.array([1, 2, 3]))
    initializer = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    default_value = tf.constant(-1)
    name = "table1"
    input_dict = {"initializer": initializer, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int keys and string values
    keys_tensor = tf.constant(np.array([1, 2, 3]))
    vals_tensor = tf.constant(np.array(['x', 'y', 'z']))
    initializer = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    default_value = tf.constant("unknown", dtype=tf.string)
    name = "table2"
    input_dict = {"initializer": initializer, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float keys and float values
    keys_tensor = tf.constant(np.array([1.1, 2.2, 3.3]))
    vals_tensor = tf.constant(np.array([4.4, 5.5, 6.6]))
    initializer = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    default_value = tf.constant(-1.0)
    name = "table3"
    input_dict = {"initializer": initializer, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty keys and values
    keys_tensor = tf.constant(np.array([]), dtype=tf.string)
    vals_tensor = tf.constant(np.array([]), dtype=tf.int32)
    initializer = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    default_value = tf.constant(-1)
    name = "table4"
    input_dict = {"initializer": initializer, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different dtypes for key and value (int64 key, float32 val)
    keys_tensor = tf.constant(np.array([1, 2, 3], dtype=np.int64))
    vals_tensor = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    initializer = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    default_value = tf.constant(-1.0, dtype=np.float32)
    name = "table5"
    input_dict = {"initializer": initializer, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Bool keys and int values
    keys_tensor = tf.constant(np.array([True, False, True]))
    vals_tensor = tf.constant(np.array([10, 20, 30]))
    initializer = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    default_value = tf.constant(-1)
    name = "table6"
    input_dict = {"initializer": initializer, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D keys and 2D values
    keys_tensor = tf.constant(np.array([['a', 'b'], ['c', 'd']]))
    vals_tensor = tf.constant(np.array([[1, 2], [3, 4]]))
    initializer = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    default_value = tf.constant(-1)
    name = "table7"
    input_dict = {"initializer": initializer, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative values in keys and values
    keys_tensor = tf.constant(np.array([-1, -2, -3]))
    vals_tensor = tf.constant(np.array([-4, -5, -6]))
    initializer = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    default_value = tf.constant(0)
    name = "table8"
    input_dict = {"initializer": initializer, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9: int64 keys and values
    keys_tensor = tf.constant(np.array([1, 2, 3], dtype=np.int64))
    vals_tensor = tf.constant(np.array([4, 5, 6], dtype=np.int64))
    initializer = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    default_value = tf.constant(0, dtype=np.int64)
    name = "table9"
    input_dict = {"initializer": initializer, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10: default value as a string
    keys_tensor = tf.constant(np.array(['a', 'b', 'c']))
    vals_tensor = tf.constant(np.array([1, 2, 3]))
    initializer = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    default_value = tf.constant("default", dtype=tf.string)
    name = "table10"
    input_dict = {"initializer": initializer, "default_value": default_value, "name": name}
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
