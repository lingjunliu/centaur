
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_group_by_window_inputs():
    """
    Generates a list of valid inputs for the tf.data.experimental.group_by_window function.
    """
    list_of_inputs = []

    # The API is a transformation function, which the test harness is expected to apply
    # to an input dataset. We provide the elements for this dataset under the key 'x'.
    # The value for 'x' is a tuple of numpy arrays, which tf.data.Dataset.from_tensor_slices
    # can use to create the dataset.

    # Due to the strict signature and the mutual exclusivity of `window_size` and
    # `window_size_func`, we can only reliably generate inputs for the `window_size`
    # case. For this, `window_size_func` can be set to `[]`, an empty list, which
    # satisfies its type requirement. Generating inputs for `window_size_func`
    # would require providing a tensor for `window_size` that is treated as `None`
    # by the API, which is not possible.

    # Input 1: Basic case with a single numpy array as the dataset element.
    input_dict_1 = {
        'x': (np.arange(20, dtype=np.int32),),
        'key_func': [lambda x: tf.cast(x % 2, tf.int64)],
        'reduce_func': [lambda key, ds: ds.batch(4)],
        'window_size': np.int64(4),
        'window_size_func': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Different keying logic and data types.
    input_dict_2 = {
        'x': (np.arange(30, dtype=np.int64),),
        'key_func': [lambda x: tf.cast(x // 10, tf.int64)],
        'reduce_func': [lambda key, ds: ds.batch(5)],
        'window_size': np.int64(5),
        'window_size_func': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Structured dataset elements (a tuple of two arrays).
    input_dict_3 = {
        'x': (np.arange(10, dtype=np.int64), np.arange(10, 20, dtype=np.int32)),
        'key_func': [lambda id, value: id % 3],
        'reduce_func': [lambda key, ds: ds.map(lambda id, val: val).batch(3)],
        'window_size': np.int64(3),
        'window_size_func': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Reduce function that utilizes the key.
    input_dict_4 = {
        'x': (np.arange(12, dtype=np.int32),),
        'key_func': [lambda x: tf.cast(x % 3, tf.int64)],
        'reduce_func': [lambda key, ds: ds.map(lambda x: x + tf.cast(key, x.dtype)).batch(2)],
        'window_size': np.int64(2),
        'window_size_func': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Minimal window size of 1.
    input_dict_5 = {
        'x': (np.arange(8, dtype=np.float32),),
        'key_func': [lambda x: tf.cast(tf.floor(x / 2.0), tf.int64)],
        'reduce_func': [lambda key, ds: ds.batch(1)],
        'window_size': np.int64(1),
        'window_size_func': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Using a different reduce function logic (summing elements).
    input_dict_6 = {
        'x': (np.arange(15, dtype=np.int64),),
        'key_func': [lambda x: x % 5],
        'reduce_func': [lambda key, ds: ds.reduce(np.int64(0), lambda a, b: a + b)],
        'window_size': np.int64(3),
        'window_size_func': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: Dataset with tensor elements
    input_dict_7 = {
        'x': (np.arange(24, dtype=np.int32).reshape(12, 2),),
        'key_func': [lambda t: tf.cast(tf.reduce_sum(t) % 4, tf.int64)],
        'reduce_func': [lambda key, ds: ds.batch(3)],
        'window_size': np.int64(3),
        'window_size_func': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

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
