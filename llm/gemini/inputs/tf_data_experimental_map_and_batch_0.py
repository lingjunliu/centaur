
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

# Define map functions at the top level to be pickle-able.
def _simple_map_func_single_arg(x):
    return x * 2

def _structured_map_func_single_arg(x):
    return (x, x + 1)

def _map_func_two_args(x, y):
    return x + y

def tf_data_experimental_map_and_batch_inputs():
    """
    Generates a list of valid inputs for the tf.data.experimental.map_and_batch function.
    The test harness requires the dataset source data to be provided under the key 'dataset'
    as a tuple of numpy arrays. 'map_func' must be a list containing the
    callable to satisfy the harness's pre-processing step.
    """
    list_of_inputs = []

    # Input 1: Basic case. Using 'dataset' as the key for the data source.
    input_dict_1 = {
        'dataset': (np.arange(20, dtype=np.int64),),
        'map_func': [_simple_map_func_single_arg],
        'batch_size': np.array(8, dtype=np.int64),
        'num_parallel_batches': None,
        'drop_remainder': np.array(False, dtype=np.bool_),
        'num_parallel_calls': None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: drop_remainder is True.
    input_dict_2 = {
        'dataset': (np.arange(21, dtype=np.int32),),
        'map_func': [_simple_map_func_single_arg],
        'batch_size': np.array(4, dtype=np.int64),
        'num_parallel_batches': None,
        'drop_remainder': np.array(True, dtype=np.bool_),
        'num_parallel_calls': None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Using num_parallel_batches.
    input_dict_3 = {
        'dataset': (np.arange(100, dtype=np.float32),),
        'map_func': [_simple_map_func_single_arg],
        'batch_size': np.array(16, dtype=np.int64),
        'num_parallel_batches': np.array(2, dtype=np.int64),
        'drop_remainder': np.array(False, dtype=np.bool_),
        'num_parallel_calls': None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Using num_parallel_calls.
    input_dict_4 = {
        'dataset': (np.arange(55, dtype=np.int64),),
        'map_func': [_simple_map_func_single_arg],
        'batch_size': np.array(10, dtype=np.int64),
        'num_parallel_batches': None,
        'drop_remainder': np.array(False, dtype=np.bool_),
        'num_parallel_calls': np.array(4, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Using AUTOTUNE for num_parallel_calls.
    input_dict_5 = {
        'dataset': (np.random.rand(100).astype(np.float32),),
        'map_func': [_simple_map_func_single_arg],
        'batch_size': np.array(32, dtype=np.int64),
        'num_parallel_batches': None,
        'drop_remainder': np.array(False, dtype=np.bool_),
        'num_parallel_calls': np.array(tf.data.AUTOTUNE, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Dataset with two tensors and a map function that accepts two args.
    input_dict_6 = {
        'dataset': (np.arange(50, dtype=np.int32), np.arange(50, 100, dtype=np.int32)),
        'map_func': [_map_func_two_args],
        'batch_size': np.array(10, dtype=np.int64),
        'num_parallel_batches': None,
        'drop_remainder': np.array(True, dtype=np.bool_),
        'num_parallel_calls': None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: A map_func that returns a nested structure (tuple).
    input_dict_7 = {
        'dataset': (np.arange(50, dtype=np.float64),),
        'map_func': [_structured_map_func_single_arg],
        'batch_size': np.array(20, dtype=np.int64),
        'num_parallel_batches': None,
        'drop_remainder': np.array(False, dtype=np.bool_),
        'num_parallel_calls': np.array(tf.data.AUTOTUNE, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: batch_size of 1.
    input_dict_8 = {
        'dataset': (np.arange(10, dtype=np.int64),),
        'map_func': [_simple_map_func_single_arg],
        'batch_size': np.array(1, dtype=np.int64),
        'num_parallel_batches': np.array(4, dtype=np.int64),
        'drop_remainder': np.array(False, dtype=np.bool_),
        'num_parallel_calls': None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    return list_of_inputs

generated_inputs["tf.data.experimental.map_and_batch"] = tf_data_experimental_map_and_batch_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.map_and_batch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.map_and_batch'.")

check_valid('tf.data.experimental.map_and_batch', generated_inputs['tf.data.experimental.map_and_batch'], lib="tf", suffix=0)
