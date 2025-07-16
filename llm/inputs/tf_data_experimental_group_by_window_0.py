
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_group_by_window_inputs():
    list_of_inputs = []

    # Input 1
    key_func = [lambda x: tf.cast(x % 2, tf.int64)]
    reduce_func = [lambda key, dataset: dataset.batch(10)]
    window_size = tf.constant(5, dtype=tf.int64)
    window_size_func = None
    input_dict = {
        "key_func": key_func,
        "reduce_func": reduce_func,
        "window_size": window_size,
        "window_size_func": window_size_func
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    key_func = [lambda x: tf.cast(x % 3, tf.int64)]
    reduce_func = [lambda key, dataset: dataset.batch(tf.cast(key + 1, tf.int64))]
    window_size = tf.constant(3, dtype=tf.int64)
    window_size_func = None
    input_dict = {
        "key_func": key_func,
        "reduce_func": reduce_func,
        "window_size": window_size,
        "window_size_func": window_size_func
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    key_func = [lambda x: tf.cast(x % 2, tf.int64)]
    reduce_func = [lambda key, dataset: dataset.batch(tf.cast(key + 1, tf.int64))]
    window_size = tf.constant(10, dtype=tf.int64)
    window_size_func = None
    input_dict = {
        "key_func": key_func,
        "reduce_func": reduce_func,
        "window_size": window_size,
        "window_size_func": window_size_func
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    key_func = [lambda x: tf.cast(x % 5, tf.int64)]
    reduce_func = [lambda key, dataset: dataset.batch(tf.cast(key + 1, tf.int64))]
    window_size = tf.constant(2, dtype=tf.int64)
    window_size_func = None
    input_dict = {
        "key_func": key_func,
        "reduce_func": reduce_func,
        "window_size": window_size,
        "window_size_func": window_size_func
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    key_func = [lambda x: tf.cast(x % 2, tf.int64)]
    reduce_func = [lambda key, dataset: dataset.batch(tf.cast(key + 1, tf.int64))]
    window_size = None
    window_size_func = [lambda key: tf.constant(5, dtype=tf.int64)]
    input_dict = {
        "key_func": key_func,
        "reduce_func": reduce_func,
        "window_size": window_size,
        "window_size_func": window_size_func
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    key_func = [lambda x: tf.cast(x % 3, tf.int64)]
    reduce_func = [lambda key, dataset: dataset.batch(tf.cast(key + 1, tf.int64))]
    window_size = None
    window_size_func = [lambda key: tf.cast(key + 2, tf.int64)]
    input_dict = {
        "key_func": key_func,
        "reduce_func": reduce_func,
        "window_size": window_size,
        "window_size_func": window_size_func
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    key_func = [lambda x: tf.cast(x % 4, tf.int64)]
    reduce_func = [lambda key, dataset: dataset.batch(tf.cast(key + 1, tf.int64))]
    window_size = None
    window_size_func = [lambda key: tf.constant(10, dtype=tf.int64)]
    input_dict = {
        "key_func": key_func,
        "reduce_func": reduce_func,
        "window_size": window_size,
        "window_size_func": window_size_func
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    key_func = [lambda x: tf.cast(x % 5, tf.int64)]
    reduce_func = [lambda key, dataset: dataset.batch(tf.cast(key + 1, tf.int64))]
    window_size = None
    window_size_func = [lambda key: tf.cast(key % 3 + 1, tf.int64)]
    input_dict = {
        "key_func": key_func,
        "reduce_func": reduce_func,
        "window_size": window_size,
        "window_size_func": window_size_func
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9
    key_func = [lambda x: tf.cast(x % 7, tf.int64)]
    reduce_func = [lambda key, dataset: dataset.batch(tf.cast(key + 1, tf.int64))]
    window_size = tf.constant(7, dtype=tf.int64)
    window_size_func = None
    input_dict = {
        "key_func": key_func,
        "reduce_func": reduce_func,
        "window_size": window_size,
        "window_size_func": window_size_func
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    key_func = [lambda x: tf.cast(x % 10, tf.int64)]
    reduce_func = [lambda key, dataset: dataset.batch(tf.cast(key + 1, tf.int64))]
    window_size = None
    window_size_func = [lambda key: tf.cast(key % 5 + 1, tf.int64)]
    input_dict = {
        "key_func": key_func,
        "reduce_func": reduce_func,
        "window_size": window_size,
        "window_size_func": window_size_func
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
