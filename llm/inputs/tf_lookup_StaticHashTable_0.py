
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_lookup_statichashtable_inputs():
    list_of_inputs = []

    # The error "TypeError: '<=' not supported between instances of 'str' and 'int'"
    # is caused by the testing framework's attempt to call np.min/np.max on an array
    # with mixed data types (created with dtype=object). To resolve this, all
    # 'initializer' arrays must have a single, homogenous data type for which
    # comparison operators are defined. Complex numbers are also problematic as they
    # do not support '<=' comparison. Therefore, the inputs are restricted to
    # homogenous integer, float, and string types.

    # Input 1: Int32 keys and values
    input_dict_1 = {
        'initializer': np.array([[1, 10], [2, 20], [3, 30]], dtype=np.int32),
        'default_value': np.int32(-1),
        'name': 'table_int32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Int64 keys and values with negative numbers
    input_dict_2 = {
        'initializer': np.array([[-100, -1000], [200, 2000], [-300, 3000]], dtype=np.int64),
        'default_value': np.int64(0),
        'name': 'table_int64_neg'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Float32 keys and values
    input_dict_3 = {
        'initializer': np.array([[1.5, 15.5], [2.5, 25.5]], dtype=np.float32),
        'default_value': np.float32(-1.0),
        'name': 'table_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Float64 keys and values
    input_dict_4 = {
        'initializer': np.array([[-1.0, -10.0], [3.14, 6.28]], dtype=np.float64),
        'default_value': np.float64(0.0),
        'name': 'table_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: String keys and values
    input_dict_5 = {
        'initializer': np.array([['a', 'apple'], ['b', 'banana'], ['c', 'cherry']]),
        'default_value': np.array('unknown'),
        'name': 'table_string'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Empty initializer tensor with a numeric dtype
    input_dict_6 = {
        'initializer': np.empty((0, 2), dtype=np.int32),
        'default_value': np.int32(42),
        'name': 'table_empty'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: uint8 keys and values
    input_dict_7 = {
        'initializer': np.array([[10, 100], [20, 110], [30, 120]], dtype=np.uint8),
        'default_value': np.uint8(255),
        'name': 'table_uint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: uint16 keys and values
    input_dict_8 = {
        'initializer': np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16),
        'default_value': np.uint16(65535),
        'name': 'table_uint16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Single key-value pair
    input_dict_9 = {
        'initializer': np.array([[12345, 54321]], dtype=np.int64),
        'default_value': np.int64(0),
        'name': 'table_single'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Larger initializer table
    input_dict_10 = {
        'initializer': np.arange(20, dtype=np.int32).reshape(10, 2),
        'default_value': np.int32(-1),
        'name': 'table_large'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.lookup.StaticHashTable"] = tf_lookup_statichashtable_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.lookup.StaticHashTable' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.lookup.StaticHashTable'.")

check_valid('tf.lookup.StaticHashTable', generated_inputs['tf.lookup.StaticHashTable'], lib="tf", suffix=0)
