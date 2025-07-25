
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_group_by_reducer_inputs():
    """
    Generates a list of valid inputs for tf.data.experimental.group_by_reducer.
    The inputs include a special key `dataset_tensors` which provides the data
    to create a tf.data.Dataset, as the API returns a transformation function.
    The values for 'key_func' and 'reducer' are wrapped in lists to adhere to the
    specified signature {'key_func': 'list', 'reducer': 'list'}.
    """
    list_of_inputs = []

    # Case 1: Sum of integers, grouped by even/odd
    reducer_sum_int32 = tf.data.experimental.Reducer(
        init_func=lambda: np.int32(0),
        reduce_func=lambda state, value: state + value,
        finalize_func=lambda state: state)
    input_dict_1 = {
        'key_func': [lambda x: x % 2],
        'reducer': [reducer_sum_int32],
        'dataset_tensors': np.arange(10, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: Count elements per group
    reducer_count = tf.data.experimental.Reducer(
        init_func=lambda: np.int64(0),
        reduce_func=lambda state, value: state + 1,
        finalize_func=lambda state: state)
    input_dict_2 = {
        'key_func': [lambda x: x // 10],
        'reducer': [reducer_count],
        'dataset_tensors': np.array([1, 5, 12, 15, 22, 31], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: Mean of floats, grouped by sign
    reducer_mean = tf.data.experimental.Reducer(
        init_func=lambda: (np.float32(0.0), np.float32(0.0)),
        reduce_func=lambda state, value: (state[0] + value, state[1] + 1.0),
        finalize_func=lambda state: state[0] / tf.maximum(state[1], 1.0))
    input_dict_3 = {
        'key_func': [lambda x: tf.cast(x > 0, tf.int64)],
        'reducer': [reducer_mean],
        'dataset_tensors': np.array([-1.0, 1.5, -2.0, 2.5, 3.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: Max value in a group (input is a tuple)
    reducer_max = tf.data.experimental.Reducer(
        init_func=lambda: np.iinfo(np.int32).min,
        reduce_func=lambda state, value: tf.maximum(state, value[1]),
        finalize_func=lambda state: state)
    input_dict_4 = {
        'key_func': [lambda k, v: k % 3],
        'reducer': [reducer_max],
        'dataset_tensors': (np.arange(10, dtype=np.int64), np.arange(10, 0, -1, dtype=np.int32))
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: Min value in a group (input is a tuple)
    reducer_min = tf.data.experimental.Reducer(
        init_func=lambda: np.iinfo(np.int32).max,
        reduce_func=lambda state, value: tf.minimum(state, value[1]),
        finalize_func=lambda state: state)
    input_dict_5 = {
        'key_func': [lambda k, v: k],
        'reducer': [reducer_min],
        'dataset_tensors': (np.array([0, 1, 0, 1, 0], dtype=np.int64), np.arange(5, dtype=np.int32))
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: Working with dictionary elements
    reducer_dict_sum = tf.data.experimental.Reducer(
        init_func=lambda: np.float64(0.0),
        reduce_func=lambda state, value: state + value['data'],
        finalize_func=lambda state: state)
    input_dict_6 = {
        'key_func': [lambda x: x['key']],
        'reducer': [reducer_dict_sum],
        'dataset_tensors': {'key': np.array([0, 1, 0, 1], dtype=np.int64), 'data': np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64)}
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: Summing with int64
    reducer_sum_int64 = tf.data.experimental.Reducer(
        init_func=lambda: np.int64(0),
        reduce_func=lambda state, value: state + value,
        finalize_func=lambda state: state)
    input_dict_7 = {
        'key_func': [lambda x: x % 5],
        'reducer': [reducer_sum_int64],
        'dataset_tensors': np.arange(20, dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: Complex finalize_func
    reducer_sum_and_double = tf.data.experimental.Reducer(
        init_func=lambda: np.int32(0),
        reduce_func=lambda state, value: state + value,
        finalize_func=lambda state: state * 2)
    input_dict_8 = {
        'key_func': [lambda x: x % 2],
        'reducer': [reducer_sum_and_double],
        'dataset_tensors': np.arange(5, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: All elements in one group
    reducer_count_all = tf.data.experimental.Reducer(
        init_func=lambda: np.int64(0),
        reduce_func=lambda state, value: state + 1,
        finalize_func=lambda state: state)
    input_dict_9 = {
        'key_func': [lambda x: tf.constant(0, dtype=tf.int64)],
        'reducer': [reducer_count_all],
        'dataset_tensors': np.random.rand(10).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: Multi-part state for variance calculation
    reducer_variance = tf.data.experimental.Reducer(
        init_func=lambda: (np.float32(0.0), np.float32(0.0), np.float32(0.0)),
        reduce_func=lambda state, value: (state[0] + 1.0, state[1] + value[1], state[2] + value[1]**2),
        finalize_func=lambda state: (state[2] / state[0]) - (state[1] / state[0])**2)
    input_dict_10 = {
        'key_func': [lambda k, v: k],
        'reducer': [reducer_variance],
        'dataset_tensors': (np.array([0, 1, 0, 1, 1], dtype=np.int64), np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.data.experimental.group_by_reducer"] = tf_data_experimental_group_by_reducer_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.group_by_reducer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.group_by_reducer'.")

check_valid('tf.data.experimental.group_by_reducer', generated_inputs['tf.data.experimental.group_by_reducer'], lib="tf", suffix=0)
