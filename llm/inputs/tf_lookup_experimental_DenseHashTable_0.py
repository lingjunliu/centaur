
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_lookup_experimental_DenseHashTable_inputs():
    """
    Generates a list of valid inputs for tf.lookup.experimental.DenseHashTable.
    """
    list_of_inputs = []

    # Based on the error log, the supported key_dtypes on CPU are
    # DT_STRING, DT_INT64, DT_INT32. Floats are not supported for keys.

    # Input 1: Basic string keys and int64 values (like the documentation)
    input_dict_1 = {
        'key_dtype': np.object_,
        'value_dtype': np.int64,
        'default_value': np.array(-1, dtype=np.int64),
        'empty_key': np.array('', dtype=np.object_),
        'deleted_key': np.array('$', dtype=np.object_),
        'name': 'string_to_int64_table'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: int32 keys and float32 values
    input_dict_2 = {
        'key_dtype': np.int32,
        'value_dtype': np.float32,
        'default_value': np.array(-99.9, dtype=np.float32),
        'empty_key': np.array(np.iinfo(np.int32).max, dtype=np.int32),
        'deleted_key': np.array(np.iinfo(np.int32).min, dtype=np.int32),
        'name': 'int32_to_float32_table'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: int64 keys and int64 values with large special keys
    input_dict_3 = {
        'key_dtype': np.int64,
        'value_dtype': np.int64,
        'default_value': np.array(0, dtype=np.int64),
        'empty_key': np.array(np.iinfo(np.int64).max, dtype=np.int64),
        'deleted_key': np.array(np.iinfo(np.int64).min, dtype=np.int64),
        'name': 'int64_to_int64_table'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: string keys and float32 values
    input_dict_4 = {
        'key_dtype': np.object_,
        'value_dtype': np.float32,
        'default_value': np.array(0.0, dtype=np.float32),
        'empty_key': np.array('EMPTY', dtype=np.object_),
        'deleted_key': np.array('DELETED', dtype=np.object_),
        'name': 'string_to_float32_table'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: string keys and bool values
    input_dict_5 = {
        'key_dtype': np.object_,
        'value_dtype': np.bool_,
        'default_value': np.array(False, dtype=np.bool_),
        'empty_key': np.array('__EMPTY__', dtype=np.object_),
        'deleted_key': np.array('__DELETED__', dtype=np.object_),
        'name': 'string_to_bool_table'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: int64 keys and bool values
    input_dict_6 = {
        'key_dtype': np.int64,
        'value_dtype': np.bool_,
        'default_value': np.array(False, dtype=np.bool_),
        'empty_key': np.array(-1, dtype=np.int64),
        'deleted_key': np.array(-2, dtype=np.int64),
        'name': 'int64_to_bool_table'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: int32 keys and int32 values
    input_dict_7 = {
        'key_dtype': np.int32,
        'value_dtype': np.int32,
        'default_value': np.array(404, dtype=np.int32),
        'empty_key': np.array(-1, dtype=np.int32),
        'deleted_key': np.array(-2, dtype=np.int32),
        'name': 'int32_to_int32_table'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: string keys and float64 (double) values
    input_dict_8 = {
        'key_dtype': np.object_,
        'value_dtype': np.float64,
        'default_value': np.array(-1.0, dtype=np.float64),
        'empty_key': np.array('e_k', dtype=np.object_),
        'deleted_key': np.array('d_k', dtype=np.object_),
        'name': 'string_to_float64_table'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: int64 keys and float32 values
    input_dict_9 = {
        'key_dtype': np.int64,
        'value_dtype': np.float32,
        'default_value': np.array(np.nan, dtype=np.float32),
        'empty_key': np.array(0, dtype=np.int64),
        'deleted_key': np.array(1, dtype=np.int64),
        'name': 'int64_to_float32_table'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: int64 keys and float64 (double) values
    input_dict_10 = {
        'key_dtype': np.int64,
        'value_dtype': np.float64,
        'default_value': np.array(0.0, dtype=np.float64),
        'empty_key': np.array(123456789, dtype=np.int64),
        'deleted_key': np.array(987654321, dtype=np.int64),
        'name': 'int64_to_float64_table'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: string key, int32 value with a valid name.
    # The 'name' parameter cannot be None.
    input_dict_11 = {
        'key_dtype': np.object_,
        'value_dtype': np.int32,
        'default_value': np.array(0, dtype=np.int32),
        'empty_key': np.array('empty_string_key', dtype=np.object_),
        'deleted_key': np.array('deleted_string_key', dtype=np.object_),
        'name': 'a_valid_name'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.lookup.experimental.DenseHashTable"] = tf_lookup_experimental_DenseHashTable_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.lookup.experimental.DenseHashTable' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.lookup.experimental.DenseHashTable'.")

check_valid('tf.lookup.experimental.DenseHashTable', generated_inputs['tf.lookup.experimental.DenseHashTable'], lib="tf", suffix=0)
