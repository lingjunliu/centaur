
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_lookup_experimental_densehashtable_inputs():
    list_of_inputs = []

    # Input 1
    key_dtype = np.dtype('bytes')
    value_dtype = np.dtype('int64')
    default_value = np.array(-1, dtype=np.int64)
    empty_key = np.array(b"", dtype=np.dtype('bytes'))
    deleted_key = np.array(b"$", dtype=np.dtype('bytes'))
    name = "table1"
    input_dict = {"key_dtype": key_dtype, "value_dtype": value_dtype, "default_value": default_value, "empty_key": empty_key, "deleted_key": deleted_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    key_dtype = np.dtype('int32')
    value_dtype = np.dtype('float32')
    default_value = np.array(0.0, dtype=np.float32)
    empty_key = np.array(-1, dtype=np.int32)
    deleted_key = np.array(-2, dtype=np.int32)
    name = "table2"
    input_dict = {"key_dtype": key_dtype, "value_dtype": value_dtype, "default_value": default_value, "empty_key": empty_key, "deleted_key": deleted_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    key_dtype = np.dtype('int64')
    value_dtype = np.dtype('bytes')
    default_value = np.array(b"unknown", dtype=np.dtype('bytes'))
    empty_key = np.array(-999, dtype=np.int64)
    deleted_key = np.array(-888, dtype=np.int64)
    name = "table3"
    input_dict = {"key_dtype": key_dtype, "value_dtype": value_dtype, "default_value": default_value, "empty_key": empty_key, "deleted_key": deleted_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    key_dtype = np.dtype('float32')
    value_dtype = np.dtype('int32')
    default_value = np.array(-10, dtype=np.int32)
    empty_key = np.array(-1.0, dtype=np.float32)
    deleted_key = np.array(-2.0, dtype=np.float32)
    name = "table4"
    input_dict = {"key_dtype": key_dtype, "value_dtype": value_dtype, "default_value": default_value, "empty_key": empty_key, "deleted_key": deleted_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    key_dtype = np.dtype('bytes')
    value_dtype = np.dtype('float64')
    default_value = np.array(0.0, dtype=np.float64)
    empty_key = np.array(b"EMPTY", dtype=np.dtype('bytes'))
    deleted_key = np.array(b"DELETED", dtype=np.dtype('bytes'))
    name = "table5"
    input_dict = {"key_dtype": key_dtype, "value_dtype": value_dtype, "default_value": default_value, "empty_key": empty_key, "deleted_key": deleted_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    key_dtype = np.dtype('int16')
    value_dtype = np.dtype('int8')
    default_value = np.array(-1, dtype=np.int8)
    empty_key = np.array(-32768, dtype=np.int16)
    deleted_key = np.array(32767, dtype=np.int16)
    name = "table6"
    input_dict = {"key_dtype": key_dtype, "value_dtype": value_dtype, "default_value": default_value, "empty_key": empty_key, "deleted_key": deleted_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    key_dtype = np.dtype('float64')
    value_dtype = np.dtype('bytes')
    default_value = np.array(b"N/A", dtype=np.dtype('bytes'))
    empty_key = np.array(-1.7976931348623157e+308, dtype=np.float64)
    deleted_key = np.array(1.7976931348623157e+308, dtype=np.float64)
    name = "table7"
    input_dict = {"key_dtype": key_dtype, "value_dtype": value_dtype, "default_value": default_value, "empty_key": empty_key, "deleted_key": deleted_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    key_dtype = np.dtype('uint32')
    value_dtype = np.dtype('int64')
    default_value = np.array(-1, dtype=np.int64)
    empty_key = np.array(0, dtype=np.uint32)
    deleted_key = np.array(1, dtype=np.uint32)
    name = "table8"
    input_dict = {"key_dtype": key_dtype, "value_dtype": value_dtype, "default_value": default_value, "empty_key": empty_key, "deleted_key": deleted_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    key_dtype = np.dtype('int32')
    value_dtype = np.dtype('bool')
    default_value = np.array(False, dtype=np.bool_)
    empty_key = np.array(-1, dtype=np.int32)
    deleted_key = np.array(-2, dtype=np.int32)
    name = "table9"
    input_dict = {"key_dtype": key_dtype, "value_dtype": value_dtype, "default_value": default_value, "empty_key": empty_key, "deleted_key": deleted_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    key_dtype = np.dtype('int64')
    value_dtype = np.dtype('float32')
    default_value = np.array(-1.0, dtype=np.float32)
    empty_key = np.array(-100, dtype=np.int64)
    deleted_key = np.array(-200, dtype=np.int64)
    name = "table10"
    input_dict = {"key_dtype": key_dtype, "value_dtype": value_dtype, "default_value": default_value, "empty_key": empty_key, "deleted_key": deleted_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.lookup.experimental.DenseHashTable"] = tf_lookup_experimental_densehashtable_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.lookup.experimental.DenseHashTable' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.lookup.experimental.DenseHashTable'.")

check_valid('tf.lookup.experimental.DenseHashTable', generated_inputs['tf.lookup.experimental.DenseHashTable'], lib="tf", suffix=0)
