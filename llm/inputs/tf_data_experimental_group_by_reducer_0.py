
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import torch

def tf_data_experimental_group_by_reducer_inputs():
    list_of_inputs = []

    # The error indicates the test harness needs a tf.data.Dataset object to apply
    # the returned function to. This version provides a pre-constructed dataset
    # under the key 'dataset'.

    # Input 1: Basic integer dataset
    input_dict_1 = {
        'dataset': tf.data.Dataset.from_tensor_slices(np.arange(10, dtype=np.int64)),
        'key_func': [np.int64(2)],
        'reducer': [np.int64(0), np.int64(1)],
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Dataset of tuples (as 2D array)
    input_dict_2 = {
        'dataset': tf.data.Dataset.from_tensor_slices(np.array([[1, 10], [2, 20], [1, 30], [2, 40]], dtype=np.int32)),
        'key_func': [np.int64(0)],
        'reducer': [np.array([0], dtype=np.int32), np.int64(1)],
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Float dataset
    input_dict_3 = {
        'dataset': tf.data.Dataset.from_tensor_slices(np.random.rand(10, 2).astype(np.float32)),
        'key_func': [np.int64(0)],
        'reducer': [np.float32(0.0), np.float32(1.0)],
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Empty dataset
    input_dict_4 = {
        'dataset': tf.data.Dataset.from_tensor_slices(np.array([], dtype=np.int64)),
        'key_func': [np.int64(1)],
        'reducer': [np.int64(0)],
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Dataset with more dimensions
    input_dict_5 = {
        'dataset': tf.data.Dataset.from_tensor_slices(np.arange(24, dtype=np.float64).reshape((2, 3, 4))),
        'key_func': [np.int64(0)],
        'reducer': [np.float64(0.0)],
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Single element dataset
    input_dict_6 = {
        'dataset': tf.data.Dataset.from_tensor_slices(np.array([-55], dtype=np.int64)),
        'key_func': [np.int64(1)],
        'reducer': [np.int64(0)],
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Boolean data
    input_dict_7 = {
        'dataset': tf.data.Dataset.from_tensor_slices(np.array([True, False, True, False, True], dtype=np.bool_)),
        'key_func': [np.int64(1)],
        'reducer': [np.int64(0), np.int64(1)],
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: Empty lists for key_func and reducer
    input_dict_8 = {
        'dataset': tf.data.Dataset.from_tensor_slices(np.arange(5, dtype=np.int64)),
        'key_func': [],
        'reducer': [],
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Nested lists for key_func and reducer
    input_dict_9 = {
        'dataset': tf.data.Dataset.from_tensor_slices(np.arange(5, dtype=np.int64)),
        'key_func': [[np.int64(1)], [np.int64(2)]],
        'reducer': [[np.int64(0)], [np.int64(1)]],
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: Using large numbers
    input_dict_10 = {
        'dataset': tf.data.Dataset.from_tensor_slices(np.array([np.iinfo(np.int64).max, np.iinfo(np.int64).min, 0], dtype=np.int64)),
        'key_func': [np.int64(3)],
        'reducer': [np.int64(0)],
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
