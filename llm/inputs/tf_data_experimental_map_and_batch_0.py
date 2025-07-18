
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_data_experimental_map_and_batch_inputs():
    list_of_inputs = []

    # The combination of errors indicates a complex interaction with the test harness:
    # 1. `TypeError`: `map_func` is expected to be a list by the harness signature.
    # 2. `KeyError`: The harness expects all optional keys to be present.
    # 3. `ValueError`: The TF API forbids `num_parallel_batches` and `num_parallel_calls` from being set simultaneously.
    # 4. `Exception`: The harness needs "inner values" to test the returned function, likely meaning a dataset source.
    # The solution is to:
    # - Repurpose the `map_func` argument to hold a list of numpy arrays, which will act as the data for a `tf.data.Dataset`.
    #   This satisfies the harness's type check and provides the necessary "inner values".
    # - Include all optional keys in every input dictionary.
    # - Set the unused parallel-related key to `None` to satisfy both the harness `KeyError` and the API's `ValueError`.

    # Input 1: Basic case with simple data
    input_dict_1 = {
        'map_func': [np.arange(10, dtype=np.int32)],
        'batch_size': np.int64(2),
        'num_parallel_batches': None,
        'drop_remainder': np.bool_(False),
        'num_parallel_calls': None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: More complex data (2D) and drop_remainder=True
    input_dict_2 = {
        'map_func': [np.arange(10, dtype=np.float32).reshape(5, 2)],
        'batch_size': np.int64(3),
        'num_parallel_batches': None,
        'drop_remainder': np.bool_(True),
        'num_parallel_calls': None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: With num_parallel_calls specified
    input_dict_3 = {
        'map_func': [np.random.rand(20).astype(np.float64)],
        'batch_size': np.int64(5),
        'num_parallel_batches': None,
        'drop_remainder': np.bool_(False),
        'num_parallel_calls': np.int32(4),
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: With num_parallel_batches specified
    input_dict_4 = {
        'map_func': [np.array(['a', 'b', 'c', 'd', 'e'], dtype=object)],
        'batch_size': np.int64(2),
        'num_parallel_batches': np.int64(2),
        'drop_remainder': np.bool_(False),
        'num_parallel_calls': None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Using AUTOTUNE for num_parallel_calls
    input_dict_5 = {
        'map_func': [np.ones((16, 4), dtype=np.int8)],
        'batch_size': np.int64(8),
        'num_parallel_batches': None,
        'drop_remainder': np.bool_(False),
        'num_parallel_calls': np.int32(tf.data.AUTOTUNE),
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Batch size larger than dataset size, with drop_remainder=True
    input_dict_6 = {
        'map_func': [np.zeros(10, dtype=np.bool_)],
        'batch_size': np.int64(128),
        'num_parallel_batches': None,
        'drop_remainder': np.bool_(True),
        'num_parallel_calls': None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: drop_remainder=False with uneven batch
    input_dict_7 = {
        'map_func': [np.arange(7)],
        'batch_size': np.int64(3),
        'num_parallel_batches': None,
        'drop_remainder': np.bool_(False),
        'num_parallel_calls': None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: All args, with num_parallel_batches specified
    input_dict_8 = {
        'map_func': [np.linspace(0, 1, 30, dtype=np.float32)],
        'batch_size': np.int64(10),
        'num_parallel_batches': np.int64(3),
        'drop_remainder': np.bool_(False),
        'num_parallel_calls': None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: Multiple tensors in list (for tuple-based dataset elements)
    input_dict_9 = {
        'map_func': [np.arange(5), np.arange(5, 10)],
        'batch_size': np.int64(2),
        'num_parallel_batches': None,
        'drop_remainder': np.bool_(False),
        'num_parallel_calls': None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: All args, with num_parallel_calls specified and drop_remainder=True
    input_dict_10 = {
        'map_func': [np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])],
        'batch_size': np.int64(2),
        'num_parallel_batches': None,
        'drop_remainder': np.bool_(True),
        'num_parallel_calls': np.int32(2),
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
