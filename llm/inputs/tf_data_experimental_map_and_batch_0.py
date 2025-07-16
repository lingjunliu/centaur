
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
    batch_size = np.int64(4)
    num_parallel_batches = np.int64(2)
    drop_remainder = np.bool_(False)
    num_parallel_calls = np.int32(1)

    input_dict = {
        "map_func": map_func,
        "batch_size": tf.constant(np.array(batch_size), dtype=tf.int64),
        "num_parallel_batches": tf.constant(np.array(num_parallel_batches), dtype=tf.int64),
        "drop_remainder": tf.constant(np.array(drop_remainder), dtype=tf.bool),
        "num_parallel_calls": tf.constant(np.array(num_parallel_calls), dtype=tf.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    map_func = [lambda x: x + 1]
    batch_size = np.int64(8)
    num_parallel_batches = np.int64(1)
    drop_remainder = np.bool_(True)
    num_parallel_calls = np.int32(4)

    input_dict = {
        "map_func": map_func,
        "batch_size": tf.constant(np.array(batch_size), dtype=tf.int64),
        "num_parallel_batches": tf.constant(np.array(num_parallel_batches), dtype=tf.int64),
        "drop_remainder": tf.constant(np.array(drop_remainder), dtype=tf.bool),
        "num_parallel_calls": tf.constant(np.array(num_parallel_calls), dtype=tf.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    map_func = [lambda x: tf.math.sin(x)]
    batch_size = np.int64(16)
    num_parallel_batches = np.int64(4)
    drop_remainder = np.bool_(False)
    num_parallel_calls = np.int32(8)

    input_dict = {
        "map_func": map_func,
        "batch_size": tf.constant(np.array(batch_size), dtype=tf.int64),
        "num_parallel_batches": tf.constant(np.array(num_parallel_batches), dtype=tf.int64),
        "drop_remainder": tf.constant(np.array(drop_remainder), dtype=tf.bool),
        "num_parallel_calls": tf.constant(np.array(num_parallel_calls), dtype=tf.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    map_func = [lambda x: tf.math.cos(x)]
    batch_size = np.int64(32)
    num_parallel_batches = np.int64(8)
    drop_remainder = np.bool_(True)
    num_parallel_calls = np.int32(16)

    input_dict = {
        "map_func": map_func,
        "batch_size": tf.constant(np.array(batch_size), dtype=tf.int64),
        "num_parallel_batches": tf.constant(np.array(num_parallel_batches), dtype=tf.int64),
        "drop_remainder": tf.constant(np.array(drop_remainder), dtype=tf.bool),
        "num_parallel_calls": tf.constant(np.array(num_parallel_calls), dtype=tf.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    map_func = [lambda x: x**2]
    batch_size = np.int64(2)
    num_parallel_batches = np.int64(3)
    drop_remainder = np.bool_(False)
    num_parallel_calls = np.int32(1)

    input_dict = {
        "map_func": map_func,
        "batch_size": tf.constant(np.array(batch_size), dtype=tf.int64),
        "num_parallel_batches": tf.constant(np.array(num_parallel_batches), dtype=tf.int64),
        "drop_remainder": tf.constant(np.array(drop_remainder), dtype=tf.bool),
        "num_parallel_calls": tf.constant(np.array(num_parallel_calls), dtype=tf.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    map_func = [lambda x: tf.cast(x, tf.float32) / 2.0]
    batch_size = np.int64(1)
    num_parallel_batches = np.int64(5)
    drop_remainder = np.bool_(True)
    num_parallel_calls = np.int32(2)

    input_dict = {
        "map_func": map_func,
        "batch_size": tf.constant(np.array(batch_size), dtype=tf.int64),
        "num_parallel_batches": tf.constant(np.array(num_parallel_batches), dtype=tf.int64),
        "drop_remainder": tf.constant(np.array(drop_remainder), dtype=tf.bool),
        "num_parallel_calls": tf.constant(np.array(num_parallel_calls), dtype=tf.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    map_func = [lambda x: tf.sqrt(tf.cast(x, tf.float32))]
    batch_size = np.int64(5)
    num_parallel_batches = np.int64(10)
    drop_remainder = np.bool_(False)
    num_parallel_calls = np.int32(5)

    input_dict = {
        "map_func": map_func,
        "batch_size": tf.constant(np.array(batch_size), dtype=tf.int64),
        "num_parallel_batches": tf.constant(np.array(num_parallel_batches), dtype=tf.int64),
        "drop_remainder": tf.constant(np.array(drop_remainder), dtype=tf.bool),
        "num_parallel_calls": tf.constant(np.array(num_parallel_calls), dtype=tf.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    map_func = [lambda x: x % 10]
    batch_size = np.int64(7)
    num_parallel_batches = np.int64(1)
    drop_remainder = np.bool_(True)
    num_parallel_calls = np.int32(2)

    input_dict = {
        "map_func": map_func,
        "batch_size": tf.constant(np.array(batch_size), dtype=tf.int64),
        "num_parallel_batches": tf.constant(np.array(num_parallel_batches), dtype=tf.int64),
        "drop_remainder": tf.constant(np.array(drop_remainder), dtype=tf.bool),
        "num_parallel_calls": tf.constant(np.array(num_parallel_calls), dtype=tf.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    map_func = [lambda x: x - 5]
    batch_size = np.int64(3)
    num_parallel_batches = np.int64(6)
    drop_remainder = np.bool_(False)
    num_parallel_calls = np.int32(3)

    input_dict = {
        "map_func": map_func,
        "batch_size": tf.constant(np.array(batch_size), dtype=tf.int64),
        "num_parallel_batches": tf.constant(np.array(num_parallel_batches), dtype=tf.int64),
        "drop_remainder": tf.constant(np.array(drop_remainder), dtype=tf.bool),
        "num_parallel_calls": tf.constant(np.array(num_parallel_calls), dtype=tf.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    map_func = [lambda x: tf.clip_by_value(x, 0, 10)]
    batch_size = np.int64(6)
    num_parallel_batches = np.int64(12)
    drop_remainder = np.bool_(True)
    num_parallel_calls = np.int32(6)

    input_dict = {
        "map_func": map_func,
        "batch_size": tf.constant(np.array(batch_size), dtype=tf.int64),
        "num_parallel_batches": tf.constant(np.array(num_parallel_batches), dtype=tf.int64),
        "drop_remainder": tf.constant(np.array(drop_remainder), dtype=tf.bool),
        "num_parallel_calls": tf.constant(np.array(num_parallel_calls), dtype=tf.int32)
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
