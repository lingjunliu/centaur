
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_map_and_batch_inputs():
    """
    Generates a list of valid inputs for the tf.data.experimental.map_and_batch function.
    Since the API returns a transformation function, the test harness is expected to
    create a tf.data.Dataset and apply the function to it. We hypothesize that the
    harness creates the dataset using `tf.data.Dataset.from_tensor_slices` and that
    the input tensors for this creation should be provided under the key 'tensors'.
    """
    list_of_inputs = []

    map_fn_simple = lambda x: x * 2
    map_fn_structured = lambda x: (x, x + 1)
    map_fn_tuple = lambda x, y: x + y
    map_fn_identity = lambda x: x

    # Input 1: Basic case with a single tensor input.
    input_dict_1 = {
        'tensors': np.arange(100, dtype=np.int64),
        'map_func': [map_fn_simple],
        'batch_size': np.array(32, dtype=np.int64),
        'num_parallel_batches': None,
        'drop_remainder': np.array(False),
        'num_parallel_calls': None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: drop_remainder=True
    input_dict_2 = {
        'tensors': np.arange(99, dtype=np.int64),
        'map_func': [map_fn_simple],
        'batch_size': np.array(10, dtype=np.int64),
        'num_parallel_batches': None,
        'drop_remainder': np.array(True),
        'num_parallel_calls': None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: With num_parallel_batches
    input_dict_3 = {
        'tensors': np.arange(100, dtype=np.int64),
        'map_func': [map_fn_simple],
        'batch_size': np.array(8, dtype=np.int64),
        'num_parallel_batches': np.array(2, dtype=np.int64),
        'drop_remainder': np.array(False),
        'num_parallel_calls': None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: With num_parallel_calls
    input_dict_4 = {
        'tensors': np.arange(100, dtype=np.int64),
        'map_func': [map_fn_simple],
        'batch_size': np.array(16, dtype=np.int64),
        'num_parallel_batches': None,
        'drop_remainder': np.array(False),
        'num_parallel_calls': np.array(4, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: With AUTOTUNE and a structured map function
    input_dict_5 = {
        'tensors': np.arange(200, dtype=np.int64),
        'map_func': [map_fn_structured],
        'batch_size': np.array(64, dtype=np.int64),
        'num_parallel_batches': None,
        'drop_remainder': np.array(False),
        'num_parallel_calls': np.array(tf.data.AUTOTUNE, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Larger batch size and num_parallel_batches
    input_dict_6 = {
        'tensors': np.arange(500, dtype=np.int64),
        'map_func': [map_fn_structured],
        'batch_size': np.array(128, dtype=np.int64),
        'num_parallel_batches': np.array(4, dtype=np.int64),
        'drop_remainder': np.array(True),
        'num_parallel_calls': None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Smallest batch size
    input_dict_7 = {
        'tensors': np.arange(10, dtype=np.int64),
        'map_func': [map_fn_simple],
        'batch_size': np.array(1, dtype=np.int64),
        'num_parallel_batches': None,
        'drop_remainder': np.array(False),
        'num_parallel_calls': None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Dataset with tuple structure
    input_dict_8 = {
        'tensors': (np.arange(50, dtype=np.int32), np.arange(50, 100, dtype=np.int32)),
        'map_func': [map_fn_tuple],
        'batch_size': np.array(5, dtype=np.int64),
        'num_parallel_batches': None,
        'drop_remainder': np.array(False),
        'num_parallel_calls': np.array(tf.data.AUTOTUNE, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Empty dataset
    input_dict_9 = {
        'tensors': np.array([], dtype=np.float32),
        'map_func': [map_fn_identity],
        'batch_size': np.array(10, dtype=np.int64),
        'num_parallel_batches': None,
        'drop_remainder': np.array(False),
        'num_parallel_calls': None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Batch size larger than dataset, no drop remainder
    input_dict_10 = {
        'tensors': np.arange(20, dtype=np.float32),
        'map_func': [map_fn_simple],
        'batch_size': np.array(30, dtype=np.int64),
        'num_parallel_batches': None,
        'drop_remainder': np.array(False),
        'num_parallel_calls': None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

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
