
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_group_by_window_inputs():
    list_of_inputs = []

    def key_func_wrapper(x):
        return tf.cast(x % 2, tf.int64)

    def reduce_func_wrapper(key, dataset):
        return dataset.batch(tf.cast(key + 1, tf.int64))

    # Input 1
    key_func = key_func_wrapper
    reduce_func = reduce_func_wrapper
    window_size = tf.constant(5, dtype=tf.int64)
    window_size_func = None
    input_dict = {
        "key_func": [key_func],
        "reduce_func": [reduce_func],
        "window_size": window_size.numpy(),
        "window_size_func": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def key_func_wrapper_2(x):
        return tf.cast(x % 3, tf.int64)

    def reduce_func_wrapper_2(key, dataset):
        return dataset.batch(tf.cast(key * 2 + 1, tf.int64))

    # Input 2
    key_func = key_func_wrapper_2
    reduce_func = reduce_func_wrapper_2
    window_size = tf.constant(3, dtype=tf.int64)
    window_size_func = None
    input_dict = {
        "key_func": [key_func],
        "reduce_func": [reduce_func],
        "window_size": window_size.numpy(),
        "window_size_func": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def key_func_wrapper_3(x):
        return tf.cast(x % 4, tf.int64)

    def reduce_func_wrapper_3(key, dataset):
        return dataset.batch(tf.cast(key + 3, tf.int64))

    # Input 3
    key_func = key_func_wrapper_3
    reduce_func = reduce_func_wrapper_3
    window_size = tf.constant(7, dtype=tf.int64)
    window_size_func = None
    input_dict = {
        "key_func": [key_func],
        "reduce_func": [reduce_func],
        "window_size": window_size.numpy(),
        "window_size_func": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def key_func_wrapper_4(x):
        return tf.cast(x % 2, tf.int64)

    def reduce_func_wrapper_4(key, dataset):
        return dataset.batch(tf.cast(tf.maximum(1, key), tf.int64))

    # Input 4
    key_func = key_func_wrapper_4
    reduce_func = reduce_func_wrapper_4
    window_size = tf.constant(2, dtype=tf.int64)
    window_size_func = None
    input_dict = {
        "key_func": [key_func],
        "reduce_func": [reduce_func],
        "window_size": window_size.numpy(),
        "window_size_func": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def key_func_wrapper_5(x):
        return tf.cast((x * x) % 5, tf.int64)

    def reduce_func_wrapper_5(key, dataset):
        return dataset.batch(tf.cast(key + 1, tf.int64))

    # Input 5
    key_func = key_func_wrapper_5
    reduce_func = reduce_func_wrapper_5
    window_size = tf.constant(4, dtype=tf.int64)
    window_size_func = None
    input_dict = {
        "key_func": [key_func],
        "reduce_func": [reduce_func],
        "window_size": window_size.numpy(),
        "window_size_func": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    def key_func_wrapper_6(x):
        return tf.cast(x % 2, tf.int64)

    def reduce_func_wrapper_6(key, dataset):
        return dataset.batch(tf.cast(key + 1, tf.int64))

    def window_size_func_wrapper_6(key):
        return tf.cast(key + 2, tf.int64)

    # Input 6
    key_func = key_func_wrapper_6
    reduce_func = reduce_func_wrapper_6
    window_size = None
    window_size_func = window_size_func_wrapper_6
    input_dict = {
        "key_func": [key_func],
        "reduce_func": [reduce_func],
        "window_size": None,
        "window_size_func": [window_size_func]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.group_by_window"] = tf_data_experimental_group_by_window_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.group_by_window' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.group_by_window'.")

check_valid('tf.data.experimental.group_by_window', generated_inputs['tf.data.experimental.group_by_window'], lib="tf", suffix=0)
