
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_map_and_batch_inputs():
    list_of_inputs = []

    # Input 1
    map_func = lambda x: x * 2
    batch_size = np.array(4, dtype=np.int64)
    num_parallel_batches = np.array(2, dtype=np.int64)
    drop_remainder = np.array(False, dtype=np.bool_)
    num_parallel_calls = None

    input_dict = {
        "map_func": map_func,
        "batch_size": batch_size,
        "num_parallel_batches": num_parallel_batches,
        "drop_remainder": drop_remainder,
        "num_parallel_calls": num_parallel_calls,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    map_func = lambda x: x + 1
    batch_size = np.array(8, dtype=np.int64)
    num_parallel_batches = None
    drop_remainder = np.array(True, dtype=np.bool_)
    num_parallel_calls = np.array(4, dtype=np.int32)

    input_dict = {
        "map_func": map_func,
        "batch_size": batch_size,
        "num_parallel_batches": num_parallel_batches,
        "drop_remainder": drop_remainder,
        "num_parallel_calls": num_parallel_calls,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    map_func = lambda x: tf.reshape(x, [1])
    batch_size = np.array(16, dtype=np.int64)
    num_parallel_batches = np.array(4, dtype=np.int64)
    drop_remainder = np.array(False, dtype=np.bool_)
    num_parallel_calls = None

    input_dict = {
        "map_func": map_func,
        "batch_size": batch_size,
        "num_parallel_batches": num_parallel_batches,
        "drop_remainder": drop_remainder,
        "num_parallel_calls": num_parallel_calls,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    map_func = lambda x: x * 3
    batch_size = np.array(32, dtype=np.int64)
    num_parallel_batches = None
    drop_remainder = np.array(True, dtype=np.bool_)
    num_parallel_calls = np.array(16, dtype=np.int32)

    input_dict = {
        "map_func": map_func,
        "batch_size": batch_size,
        "num_parallel_batches": num_parallel_batches,
        "drop_remainder": drop_remainder,
        "num_parallel_calls": num_parallel_calls,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    map_func = lambda x: x / 2
    batch_size = np.array(2, dtype=np.int64)
    num_parallel_batches = np.array(3, dtype=np.int64)
    drop_remainder = np.array(False, dtype=np.bool_)
    num_parallel_calls = None

    input_dict = {
        "map_func": map_func,
        "batch_size": batch_size,
        "num_parallel_batches": num_parallel_batches,
        "drop_remainder": drop_remainder,
        "num_parallel_calls": num_parallel_calls,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    map_func = lambda x: x * 5
    batch_size = np.array(5, dtype=np.int64)
    num_parallel_batches = None
    drop_remainder = np.array(True, dtype=np.bool_)
    num_parallel_calls = np.array(2, dtype=np.int32)

    input_dict = {
        "map_func": map_func,
        "batch_size": batch_size,
        "num_parallel_batches": num_parallel_batches,
        "drop_remainder": drop_remainder,
        "num_parallel_calls": num_parallel_calls,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    map_func = lambda x: x - 10
    batch_size = np.array(10, dtype=np.int64)
    num_parallel_batches = np.array(5, dtype=np.int64)
    drop_remainder = np.array(False, dtype=np.bool_)
    num_parallel_calls = None

    input_dict = {
        "map_func": map_func,
        "batch_size": batch_size,
        "num_parallel_batches": num_parallel_batches,
        "drop_remainder": drop_remainder,
        "num_parallel_calls": num_parallel_calls,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    map_func = lambda x: tf.cast(x, tf.float32)
    batch_size = np.array(1, dtype=np.int64)
    num_parallel_batches = None
    drop_remainder = np.array(True, dtype=np.bool_)
    num_parallel_calls = np.array(1, dtype=np.int32)

    input_dict = {
        "map_func": map_func,
        "batch_size": batch_size,
        "num_parallel_batches": num_parallel_batches,
        "drop_remainder": drop_remainder,
        "num_parallel_calls": num_parallel_calls,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    map_func = lambda x: x ** 2
    batch_size = np.array(7, dtype=np.int64)
    num_parallel_batches = np.array(14, dtype=np.int64)
    drop_remainder = np.array(False, dtype=np.bool_)
    num_parallel_calls = None

    input_dict = {
        "map_func": map_func,
        "batch_size": batch_size,
        "num_parallel_batches": num_parallel_batches,
        "drop_remainder": drop_remainder,
        "num_parallel_calls": num_parallel_calls,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    map_func = lambda x: tf.strings.as_string(x)
    batch_size = np.array(64, dtype=np.int64)
    num_parallel_batches = None
    drop_remainder = np.array(True, dtype=np.bool_)
    num_parallel_calls = np.array(32, dtype=np.int32)

    input_dict = {
        "map_func": map_func,
        "batch_size": batch_size,
        "num_parallel_batches": num_parallel_batches,
        "drop_remainder": drop_remainder,
        "num_parallel_calls": num_parallel_calls,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    map_func = lambda x: x + 5
    batch_size = np.array(128, dtype=np.int64)
    num_parallel_batches = np.array(10, dtype=np.int64)
    drop_remainder = np.array(False, dtype=np.bool_)
    num_parallel_calls = None

    input_dict = {
        "map_func": map_func,
        "batch_size": batch_size,
        "num_parallel_batches": num_parallel_batches,
        "drop_remainder": drop_remainder,
        "num_parallel_calls": num_parallel_calls,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    map_func = lambda x: x * 1.5
    batch_size = np.array(3, dtype=np.int64)
    num_parallel_batches = None
    drop_remainder = np.array(True, dtype=np.bool_)
    num_parallel_calls = np.array(3, dtype=np.int32)

    input_dict = {
        "map_func": map_func,
        "batch_size": batch_size,
        "num_parallel_batches": num_parallel_batches,
        "drop_remainder": drop_remainder,
        "num_parallel_calls": num_parallel_calls,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.data.experimental.map_and_batch"] = tf_data_experimental_map_and_batch_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.map_and_batch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.map_and_batch'.")

check_valid('tf.data.experimental.map_and_batch', generated_inputs['tf.data.experimental.map_and_batch'], lib="tf", suffix=0)
