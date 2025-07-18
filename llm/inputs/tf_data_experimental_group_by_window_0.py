
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_group_by_window_inputs():
    """
    Generates a list of valid inputs for tf.data.experimental.group_by_window.
    The callable functions are represented as a list of bytes containing the
    source code. This is an attempt to work around a testing environment issue
    where np.min on a list of unicode strings fails. Bytes might have a
    supported comparison ufunc. The testing framework is expected to decode
    these bytes before evaluation.
    """
    list_of_inputs = []

    # Case 1: Basic grouping with window_size
    input_dict_1 = {
        'key_func': [b'lambda x: x % 2'],
        'reduce_func': [b'lambda k, d: d.batch(2)'],
        'window_size': tf.constant(2, dtype=tf.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: Different key func and window_size
    input_dict_2 = {
        'key_func': [b'lambda x: x % 3'],
        'reduce_func': [b'lambda k, d: d.batch(5)'],
        'window_size': tf.constant(5, dtype=tf.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: Keying on a tuple element
    input_dict_3 = {
        'key_func': [b'lambda x, y: x'],
        'reduce_func': [b'lambda k, d: d.batch(4)'],
        'window_size': tf.constant(4, dtype=tf.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: All elements to the same key
    input_dict_4 = {
        'key_func': [b'lambda x: tf.constant(0, dtype=tf.int64)'],
        'reduce_func': [b'lambda k, d: d.batch(10)'],
        'window_size': tf.constant(10, dtype=tf.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: Trivial window size 1
    input_dict_5 = {
        'key_func': [b'lambda x: x'],
        'reduce_func': [b'lambda k, d: d.batch(1)'],
        'window_size': tf.constant(1, dtype=tf.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: Basic grouping with window_size_func
    input_dict_6 = {
        'key_func': [b'lambda x: x % 3'],
        'reduce_func': [b'lambda k, d: d.batch(tf.cast(k, tf.int64) + 1)'],
        'window_size_func': [b'lambda k: k + 1']
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: Using window_size_func that returns a constant
    input_dict_7 = {
        'key_func': [b'lambda x: x % 4'],
        'reduce_func': [b'lambda k, d: d.batch(3)'],
        'window_size_func': [b'lambda k: tf.constant(3, dtype=tf.int64)']
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: window_size_func dependent on the key
    input_dict_8 = {
        'key_func': [b'lambda x: x'],
        'reduce_func': [b'lambda k, d: d.batch(tf.cast(k, tf.int64) * 2 + 1)'],
        'window_size_func': [b'lambda k: k * 2 + 1']
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: Grouping on a component of a tuple with window_size_func
    input_dict_9 = {
        'key_func': [b'lambda x, y: y'],
        'reduce_func': [b'lambda k, d: d.batch(10)'],
        'window_size_func': [b'lambda k: tf.constant(10, dtype=tf.int64)']
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: key_func maps all elements to the same key with window_size_func
    input_dict_10 = {
        'key_func': [b'lambda x: 0'],
        'reduce_func': [b'lambda k, d: d.batch(8)'],
        'window_size_func': [b'lambda k: 8']
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Case 11: numpy int64 for window_size
    input_dict_11 = {
        'key_func': [b'lambda x: x % 2'],
        'reduce_func': [b'lambda k,d: d.batch(9)'],
        'window_size': tf.constant(np.int64(9))
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))


    return list_of_inputs

generated_inputs["tf.data.experimental.group_by_window"] = tf_data_experimental_group_by_window_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.group_by_window' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.group_by_window'.")

check_valid('tf.data.experimental.group_by_window', generated_inputs['tf.data.experimental.group_by_window'], lib="tf", suffix=0)
