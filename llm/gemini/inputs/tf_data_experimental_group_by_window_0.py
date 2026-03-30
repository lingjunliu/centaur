
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_data_experimental_group_by_window_inputs():
    list_of_inputs = []

    # Input 1
    def key_func_1(x):
        return tf.cast(x % 2, tf.int64)

    def reduce_func_1(key, dataset):
        return dataset.batch(10)

    window_size_1 = tf.constant(5, dtype=tf.int64)

    input_dict = {
        "key_func": [key_func_1],
        "reduce_func": [reduce_func_1],
        "window_size": window_size_1,
        "window_size_func": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    def key_func_2(x):
        return tf.cast(x % 3, tf.int64)

    def reduce_func_2(key, dataset):
        return dataset.batch(1)

    window_size_2 = tf.constant(2, dtype=tf.int64)

    input_dict = {
        "key_func": [key_func_2],
        "reduce_func": [reduce_func_2],
        "window_size": window_size_2,
        "window_size_func": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    def key_func_3(x):
        return tf.cast(x % 4, tf.int64)

    def reduce_func_3(key, dataset):
        return dataset.batch(tf.cast(key + 1, tf.int64))

    window_size_3 = tf.constant(3, dtype=tf.int64)

    input_dict = {
        "key_func": [key_func_3],
        "reduce_func": [reduce_func_3],
        "window_size": window_size_3,
        "window_size_func": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
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


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.data.experimental.group_by_window', generated_inputs['tf.data.experimental.group_by_window'], lib="tf", suffix=0)
