
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_lookup_KeyValueTensorInitializer_inputs():
    list_of_inputs = []

    # Input 1
    keys = np.array(['a', 'b', 'c'], dtype=np.unicode_)
    values = np.array([1, 2, 3])
    key_dtype = np.dtype('U1')
    value_dtype = np.dtype('int32')
    name = "table_init_1"
    input_dict = {"keys": keys, "values": values, "key_dtype": key_dtype, "value_dtype": value_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    keys = np.array([1, 2, 3])
    values = np.array([4.0, 5.0, 6.0])
    key_dtype = np.dtype('int32')
    value_dtype = np.dtype('float32')
    name = "table_init_2"
    input_dict = {"keys": keys, "values": values, "key_dtype": key_dtype, "value_dtype": value_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    keys = np.array(['x', 'y', 'z'], dtype=np.unicode_)
    values = np.array([True, False, True])
    key_dtype = np.dtype('U1')
    value_dtype = np.dtype('bool')
    name = "table_init_3"
    input_dict = {"keys": keys, "values": values, "key_dtype": key_dtype, "value_dtype": value_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    keys = np.array([1.0, 2.0, 3.0])
    values = np.array(['p', 'q', 'r'], dtype=np.unicode_)
    key_dtype = np.dtype('float32')
    value_dtype = np.dtype('U1')
    name = "table_init_4"
    input_dict = {"keys": keys, "values": values, "key_dtype": key_dtype, "value_dtype": value_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    keys = np.array([10, 20, 30])
    values = np.array([1.1, 2.2, 3.3])
    key_dtype = np.dtype('int32')
    value_dtype = np.dtype('float64')
    name = "table_init_5"
    input_dict = {"keys": keys, "values": values, "key_dtype": key_dtype, "value_dtype": value_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    keys = np.array(['aa', 'bb', 'cc'], dtype=np.unicode_)
    values = np.array([100, 200, 300])
    key_dtype = np.dtype('U2')
    value_dtype = np.dtype('int64')
    name = "table_init_6"
    input_dict = {"keys": keys, "values": values, "key_dtype": key_dtype, "value_dtype": value_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    keys = np.array([-1, -2, -3])
    values = np.array([0, 1, 2])
    key_dtype = np.dtype('int32')
    value_dtype = np.dtype('int32')
    name = "table_init_7"
    input_dict = {"keys": keys, "values": values, "key_dtype": key_dtype, "value_dtype": value_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    keys = np.array([0, 1])
    values = np.array([['a', 'b'], ['c', 'd']], dtype=np.unicode_)
    key_dtype = np.dtype('int32')
    value_dtype = np.dtype('U1')
    name = "table_init_8"
    input_dict = {"keys": keys, "values": values, "key_dtype": key_dtype, "value_dtype": value_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    keys = np.array([['a', 'b'], ['c', 'd']], dtype=np.unicode_)
    values = np.array([1, 2])
    key_dtype = np.dtype('U1')
    value_dtype = np.dtype('int32')
    name = "table_init_9"
    input_dict = {"keys": keys, "values": values, "key_dtype": key_dtype, "value_dtype": value_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    keys = np.array([1])
    values = np.array([10])
    key_dtype = np.dtype('int32')
    value_dtype = np.dtype('int32')
    name = "table_init_10"
    input_dict = {"keys": keys, "values": values, "key_dtype": key_dtype, "value_dtype": value_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    keys = np.array([1,2,3], dtype=np.int64)
    values = np.array([4,5,6], dtype=np.int64)
    key_dtype = np.dtype('int64')
    value_dtype = np.dtype('int64')
    name = "table_init_11"
    input_dict = {"keys": keys, "values": values, "key_dtype": key_dtype, "value_dtype": value_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    keys = np.array([1.0, 2.0], dtype=np.float64)
    values = np.array([['a', 'b'], ['c', 'd']], dtype=np.unicode_)
    key_dtype = np.dtype('float64')
    value_dtype = np.dtype('U1')
    name = "table_init_12"
    input_dict = {"keys": keys, "values": values, "key_dtype": key_dtype, "value_dtype": value_dtype, "name": name}
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
