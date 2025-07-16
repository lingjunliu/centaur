
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_map_and_batch_inputs():
    list_of_inputs = []

    # Input 1
    map_func = [lambda x: x * 2]
    batch_size = tf.constant(4, dtype=tf.int64)
    num_parallel_batches = tf.constant(2, dtype=tf.int64)
    drop_remainder = tf.constant(False, dtype=tf.bool)
    num_parallel_calls = tf.constant(2, dtype=tf.int32)
    input_dict = {
        "map_func": map_func,
        "batch_size": batch_size,
        "num_parallel_batches": num_parallel_batches,
        "drop_remainder": drop_remainder,
        "num_parallel_calls": num_parallel_calls
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    map_func = [lambda x: x + 1]
    batch_size = tf.constant(8, dtype=tf.int64)
    num_parallel_batches = tf.constant(1, dtype=tf.int64)
    drop_remainder = tf.constant(True, dtype=tf.bool)
    num_parallel_calls = tf.constant(4, dtype=tf.int32)
    input_dict = {
        "map_func": map_func,
        "batch_size": batch_size,
        "num_parallel_batches": num_parallel_batches,
        "drop_remainder": drop_remainder,
        "num_parallel_calls": num_parallel_calls
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    map_func = [lambda x: x + 5]
    batch_size = tf.constant(16, dtype=tf.int64)
    num_parallel_batches = tf.constant(1, dtype=tf.int64)
    drop_remainder = tf.constant(True, dtype=tf.bool)
    num_parallel_calls = tf.constant(4, dtype=tf.int32)
    input_dict = {
        "map_func": map_func,
        "batch_size": batch_size,
        "num_parallel_batches": num_parallel_batches,
        "drop_remainder": drop_remainder,
        "num_parallel_calls": num_parallel_calls
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    map_func = [lambda x: x * x]
    batch_size = tf.constant(2, dtype=tf.int64)
    num_parallel_batches = tf.constant(1, dtype=tf.int64)
    drop_remainder = tf.constant(False, dtype=tf.bool)
    num_parallel_calls = tf.constant(2, dtype=tf.int32)
    input_dict = {
        "map_func": map_func,
        "batch_size": batch_size,
        "num_parallel_batches": num_parallel_batches,
        "drop_remainder": drop_remainder,
        "num_parallel_calls": num_parallel_calls
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    map_func = [lambda x: tf.cast(x, tf.float32) / 255.0]
    batch_size = tf.constant(4, dtype=tf.int64)
    num_parallel_batches = tf.constant(1, dtype=tf.int64)
    drop_remainder = tf.constant(False, dtype=tf.bool)
    num_parallel_calls = tf.constant(2, dtype=tf.int32)
    input_dict = {
        "map_func": map_func,
        "batch_size": batch_size,
        "num_parallel_batches": num_parallel_batches,
        "drop_remainder": drop_remainder,
        "num_parallel_calls": num_parallel_calls
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    map_func = [lambda x: x * 1000]
    batch_size = tf.constant(4, dtype=tf.int64)
    num_parallel_batches = tf.constant(1, dtype=tf.int64)
    drop_remainder = tf.constant(False, dtype=tf.bool)
    num_parallel_calls = tf.constant(2, dtype=tf.int32)
    input_dict = {
        "map_func": map_func,
        "batch_size": batch_size,
        "num_parallel_batches": num_parallel_batches,
        "drop_remainder": drop_remainder,
        "num_parallel_calls": num_parallel_calls
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    map_func = [lambda x: tf.math.sqrt(tf.cast(x, tf.float32))]
    batch_size = tf.constant(4, dtype=tf.int64)
    num_parallel_batches = tf.constant(1, dtype=tf.int64)
    drop_remainder = tf.constant(False, dtype=tf.bool)
    num_parallel_calls = tf.constant(2, dtype=tf.int32)
    input_dict = {
        "map_func": map_func,
        "batch_size": batch_size,
        "num_parallel_batches": num_parallel_batches,
        "drop_remainder": drop_remainder,
        "num_parallel_calls": num_parallel_calls
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    map_func = [lambda x: tf.clip_by_value(x, 0, 1)]
    batch_size = tf.constant(4, dtype=tf.int64)
    num_parallel_batches = tf.constant(1, dtype=tf.int64)
    drop_remainder = tf.constant(False, dtype=tf.bool)
    num_parallel_calls = tf.constant(2, dtype=tf.int32)
    input_dict = {
        "map_func": map_func,
        "batch_size": batch_size,
        "num_parallel_batches": num_parallel_batches,
        "drop_remainder": drop_remainder,
        "num_parallel_calls": num_parallel_calls
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    map_func = [lambda x: tf.math.sin(x)]
    batch_size = tf.constant(4, dtype=tf.int64)
    num_parallel_batches = tf.constant(1, dtype=tf.int64)
    drop_remainder = tf.constant(False, dtype=tf.bool)
    num_parallel_calls = tf.constant(2, dtype=tf.int32)
    input_dict = {
        "map_func": map_func,
        "batch_size": batch_size,
        "num_parallel_batches": num_parallel_batches,
        "drop_remainder": drop_remainder,
        "num_parallel_calls": num_parallel_calls
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    map_func = [lambda x: x - 1]
    batch_size = tf.constant(4, dtype=tf.int64)
    num_parallel_batches = tf.constant(1, dtype=tf.int64)
    drop_remainder = tf.constant(False, dtype=tf.bool)
    num_parallel_calls = tf.constant(2, dtype=tf.int32)
    input_dict = {
        "map_func": map_func,
        "batch_size": batch_size,
        "num_parallel_batches": num_parallel_batches,
        "drop_remainder": drop_remainder,
        "num_parallel_calls": num_parallel_calls
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.map_and_batch"] = tf_data_experimental_map_and_batch_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.map_and_batch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.map_and_batch'.")

check_valid('tf.data.experimental.map_and_batch', generated_inputs['tf.data.experimental.map_and_batch'], lib="tf", suffix=0)
