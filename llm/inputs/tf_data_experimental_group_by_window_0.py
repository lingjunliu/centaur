
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_group_by_window_inputs():
    """
    Generates a list of valid inputs for the tf.data.experimental.group_by_window function.
    This function returns a transformation function, which is then applied to a dataset.
    The test harness is expected to create a dataset from the 'self' key.
    """
    list_of_inputs = []

    # All inputs will use `window_size` and provide an empty list for `window_size_func`
    # to strictly adhere to the provided signature and avoid None values.
    # 'self' is provided as the data for the dataset.
    # 'window_size' is a numpy array scalar to match 'tensor' type.
    # 'key_func' and 'reduce_func' are wrapped in lists.
    # 'window_size_func' is an empty list.

    # Input 1
    key_func_1 = lambda x: x % 2
    reduce_func_1 = lambda key, ds: ds.batch(2)
    input_dict_1 = {
        'self': np.arange(10, dtype=np.int64),
        'key_func': [key_func_1],
        'reduce_func': [reduce_func_1],
        'window_size': np.array(2, dtype=np.int64),
        'window_size_func': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2
    key_func_2 = lambda x: tf.constant(0, dtype=tf.int64)
    reduce_func_2 = lambda key, ds: ds.batch(4)
    input_dict_2 = {
        'self': np.arange(8, dtype=np.int64),
        'key_func': [key_func_2],
        'reduce_func': [reduce_func_2],
        'window_size': np.array(4, dtype=np.int64),
        'window_size_func': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3
    key_func_3 = lambda x: x
    reduce_func_3 = lambda key, ds: ds
    input_dict_3 = {
        'self': np.arange(5, dtype=np.int64),
        'key_func': [key_func_3],
        'reduce_func': [reduce_func_3],
        'window_size': np.array(1, dtype=np.int64),
        'window_size_func': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4
    key_func_4 = lambda x: x // 5
    reduce_func_4 = lambda key, ds: ds.batch(5)
    input_dict_4 = {
        'self': np.arange(20, dtype=np.int64),
        'key_func': [key_func_4],
        'reduce_func': [reduce_func_4],
        'window_size': np.array(5, dtype=np.int64),
        'window_size_func': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5
    key_func_5 = lambda x: tf.cast(x > 10, dtype=tf.int64)
    reduce_func_5 = lambda key, ds: ds.batch(8)
    input_dict_5 = {
        'self': np.arange(20, dtype=np.int64),
        'key_func': [key_func_5],
        'reduce_func': [reduce_func_5],
        'window_size': np.array(8, dtype=np.int64),
        'window_size_func': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Tuple data for 'self'
    key_func_6 = lambda x, y: x % 3
    reduce_func_6 = lambda key, ds: ds.batch(4)
    input_dict_6 = {
        'self': (np.arange(12, dtype=np.int64), np.random.randint(0, 100, size=(12,), dtype=np.int64)),
        'key_func': [key_func_6],
        'reduce_func': [reduce_func_6],
        'window_size': np.array(4, dtype=np.int64),
        'window_size_func': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Dict data for 'self'
    key_func_7 = lambda d: d['id'] % 2
    reduce_func_7 = lambda key, ds: ds.padded_batch(5, padded_shapes={'id':[], 'val':[]})
    input_dict_7 = {
        'self': {'id': np.arange(10, dtype=np.int64), 'val': np.arange(100, 110, dtype=np.int64)},
        'key_func': [key_func_7],
        'reduce_func': [reduce_func_7],
        'window_size': np.array(5, dtype=np.int64),
        'window_size_func': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8
    key_func_8 = lambda x: x % 2
    reduce_func_8 = lambda key, ds: ds.batch(1)
    input_dict_8 = {
        'self': np.array([1, 1, 2, 2, 1, 1], dtype=np.int64),
        'key_func': [key_func_8],
        'reduce_func': [reduce_func_8],
        'window_size': np.array(1, dtype=np.int64),
        'window_size_func': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9
    key_func_9 = lambda x: x % 3
    reduce_func_9 = lambda key, ds: ds.batch(6)
    input_dict_9 = {
        'self': np.arange(30, dtype=np.int64),
        'key_func': [key_func_9],
        'reduce_func': [reduce_func_9],
        'window_size': np.array(6, dtype=np.int64),
        'window_size_func': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10
    key_func_10 = lambda x: x // 100
    reduce_func_10 = lambda key, ds: ds.batch(10)
    input_dict_10 = {
        'self': np.arange(15, dtype=np.int64),
        'key_func': [key_func_10],
        'reduce_func': [reduce_func_10],
        'window_size': np.array(10, dtype=np.int64),
        'window_size_func': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

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
