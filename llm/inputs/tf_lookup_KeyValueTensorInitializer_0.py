
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_lookup_KeyValueTensorInitializer_inputs():
    list_of_inputs = []

    # Input 1
    keys = np.array(['a', 'b', 'c'], dtype=np.object_)
    values = np.array([1, 2, 3], dtype=np.int32)
    key_dtype = np.string_
    value_dtype = np.int32
    name = "table_init_1"
    input_dict = {
        "keys": keys,
        "values": values,
        "key_dtype": key_dtype,
        "value_dtype": value_dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    keys = np.array([1, 2, 3], dtype=np.int32)
    values = np.array(['x', 'y', 'z'], dtype=np.object_)
    key_dtype = np.int32
    value_dtype = np.string_
    name = "table_init_2"
    input_dict = {
        "keys": keys,
        "values": values,
        "key_dtype": key_dtype,
        "value_dtype": value_dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    keys = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    values = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    key_dtype = np.float32
    value_dtype = np.float32
    name = "table_init_3"
    input_dict = {
        "keys": keys,
        "values": values,
        "key_dtype": key_dtype,
        "value_dtype": value_dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    keys = np.array(['a', 'b', 'c'], dtype=np.object_)
    values = np.array([1, 2, 3], dtype=np.int64)
    key_dtype = np.string_
    value_dtype = np.int64
    name = "table_init_4"
    input_dict = {
        "keys": keys,
        "values": values,
        "key_dtype": key_dtype,
        "value_dtype": value_dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    keys = np.array([1, 2, 3], dtype=np.int64)
    values = np.array(['x', 'y', 'z'], dtype=np.object_)
    key_dtype = np.int64
    value_dtype = np.string_
    name = "table_init_5"
    input_dict = {
        "keys": keys,
        "values": values,
        "key_dtype": key_dtype,
        "value_dtype": value_dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    keys = np.array([True, False, True], dtype=np.bool_)
    values = np.array([1, 0, 1], dtype=np.int32)
    key_dtype = np.bool_
    value_dtype = np.int32
    name = "table_init_6"
    input_dict = {
        "keys": keys,
        "values": values,
        "key_dtype": key_dtype,
        "value_dtype": value_dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    keys = np.array(['a', 'b', 'c'], dtype=np.object_)
    values = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    key_dtype = np.string_
    value_dtype = np.float64
    name = "table_init_7"
    input_dict = {
        "keys": keys,
        "values": values,
        "key_dtype": key_dtype,
        "value_dtype": value_dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    keys = np.array([1, 2, 3], dtype=np.int32)
    values = np.array([True, False, True], dtype=np.bool_)
    key_dtype = np.int32
    value_dtype = np.bool_
    name = "table_init_8"
    input_dict = {
        "keys": keys,
        "values": values,
        "key_dtype": key_dtype,
        "value_dtype": value_dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (empty arrays)
    keys = np.array([], dtype=np.object_)
    values = np.array([], dtype=np.int32)
    key_dtype = np.string_
    value_dtype = np.int32
    name = "table_init_9"
    input_dict = {
        "keys": keys,
        "values": values,
        "key_dtype": key_dtype,
        "value_dtype": value_dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (multidimensional arrays - same shape)
    keys = np.array([['a', 'b'], ['c', 'd']], dtype=np.object_)
    values = np.array([[1, 2], [3, 4]], dtype=np.int32)
    key_dtype = np.string_
    value_dtype = np.int32
    name = "table_init_10"
    input_dict = {
        "keys": keys,
        "values": values,
        "key_dtype": key_dtype,
        "value_dtype": value_dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.lookup.KeyValueTensorInitializer"] = tf_lookup_KeyValueTensorInitializer_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.lookup.KeyValueTensorInitializer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.lookup.KeyValueTensorInitializer'.")

check_valid('tf.lookup.KeyValueTensorInitializer', generated_inputs['tf.lookup.KeyValueTensorInitializer'], lib="tf", suffix=0)
