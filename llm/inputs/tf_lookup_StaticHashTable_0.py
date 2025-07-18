
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy


def get_tf_lookup_statichashtable_inputs():

    class PatchedKeyValueTensorInitializer(tf.lookup.KeyValueTensorInitializer):
        def __init__(self, keys, values, key_dtype=None, value_dtype=None, name=None):
            super().__init__(keys, values, key_dtype, value_dtype, name)
            self.shape = self._keys.shape
            self.size = tf.size(self._keys).numpy()
            if self._keys.dtype == tf.string:
                self.dtype = tf.int64
            else:
                self.dtype = self._keys.dtype

    list_of_inputs = []

    # Input 1: Basic string keys to int32 values
    keys1 = np.array(['a', 'b', 'c'])
    vals1 = np.array([7, 8, 9], dtype=np.int32)
    default1 = np.array(-1, dtype=np.int32)
    input_dict1 = {
        'initializer': PatchedKeyValueTensorInitializer(
            keys=tf.constant(keys1, dtype=tf.string),
            values=tf.constant(vals1)
        ),
        'default_value': default1,
        'name': 'string_to_int_table'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Basic int64 keys to float32 values
    keys2 = np.array([10, 20, 30], dtype=np.int64)
    vals2 = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    default2 = np.array(-1.0, dtype=np.float32)
    input_dict2 = {
        'initializer': PatchedKeyValueTensorInitializer(
            keys=tf.constant(keys2),
            values=tf.constant(vals2)
        ),
        'default_value': default2,
        'name': 'int_to_float_table'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: int32 keys to string values
    keys3 = np.array([1, 2, 3], dtype=np.int32)
    vals3 = np.array(['one', 'two', 'three'])
    default3 = np.array('<UNK>')
    input_dict3 = {
        'initializer': PatchedKeyValueTensorInitializer(
            keys=tf.constant(keys3),
            values=tf.constant(vals3, dtype=tf.string)
        ),
        'default_value': default3,
        'name': 'int_to_string_table'
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Empty keys and values
    keys4 = np.array([], dtype=np.int64)
    vals4 = np.array([], dtype=np.int32)
    default4 = np.array(42, dtype=np.int32)
    input_dict4 = {
        'initializer': PatchedKeyValueTensorInitializer(
            keys=tf.constant(keys4),
            values=tf.constant(vals4)
        ),
        'default_value': default4,
        'name': 'empty_table'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: String keys to int64 values
    keys5 = np.array(['large', 'small'])
    vals5 = np.array([2**40, -(2**40)], dtype=np.int64)
    default5 = np.array(0, dtype=np.int64)
    input_dict5 = {
        'initializer': PatchedKeyValueTensorInitializer(
            keys=tf.constant(keys5, dtype=tf.string),
            values=tf.constant(vals5)
        ),
        'default_value': default5,
        'name': 'string_to_int64_table'
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Negative integer keys to negative integer values
    keys6 = np.array([-10, -20, -30], dtype=np.int64)
    vals6 = np.array([-1, -2, -3], dtype=np.int32)
    default6 = np.array(0, dtype=np.int32)
    input_dict6 = {
        'initializer': PatchedKeyValueTensorInitializer(
            keys=tf.constant(keys6),
            values=tf.constant(vals6)
        ),
        'default_value': default6,
        'name': 'negative_keys_table'
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: No name parameter, int keys to bool values
    keys7 = np.array([100, 200], dtype=np.int64)
    vals7 = np.array([True, False], dtype=bool)
    default7 = np.array(False, dtype=bool)
    input_dict7 = {
        'initializer': PatchedKeyValueTensorInitializer(
            keys=tf.constant(keys7),
            values=tf.constant(vals7)
        ),
        'default_value': default7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: String keys to float64 values
    keys8 = np.array(['pi', 'e'])
    vals8 = np.array([3.1415926535, 2.7182818284], dtype=np.float64)
    default8 = np.array(0.0, dtype=np.float64)
    input_dict8 = {
        'initializer': PatchedKeyValueTensorInitializer(
            keys=tf.constant(keys8, dtype=tf.string),
            values=tf.constant(vals8)
        ),
        'default_value': default8,
        'name': 'float64_table'
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Unicode characters in keys and values
    keys9 = np.array(['α', 'β', 'γ'])
    vals9 = np.array(['Α', 'Β', 'Γ'])
    default9 = np.array('Ω')
    input_dict9 = {
        'initializer': PatchedKeyValueTensorInitializer(
            keys=tf.constant(keys9, dtype=tf.string),
            values=tf.constant(vals9, dtype=tf.string)
        ),
        'default_value': default9,
        'name': 'unicode_table'
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Byte string keys and values
    keys10 = np.array([b'key1', b'key2'])
    vals10 = np.array([b'val1', b'val2'])
    default10 = np.array(b'default')
    input_dict10 = {
        'initializer': PatchedKeyValueTensorInitializer(
            keys=tf.constant(keys10, dtype=tf.string),
            values=tf.constant(vals10, dtype=tf.string)
        ),
        'default_value': default10,
        'name': 'bytes_table'
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.lookup.StaticHashTable"] = get_tf_lookup_statichashtable_inputs()

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
