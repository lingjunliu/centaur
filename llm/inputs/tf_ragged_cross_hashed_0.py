
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_ragged_cross_hashed_inputs():
    """
    Generates a list of valid inputs for the tf.ragged.cross_hashed function.
    To avoid issues with the test harness, all generated inputs contain a list
    with only a single tensor.
    """
    list_of_inputs = []

    # Input 1: Basic RaggedTensor of int64
    rt1 = tf.ragged.constant([[10], [20, 30]], dtype=tf.int64)
    input_dict_1 = {
        'inputs': np.array([rt1], dtype=object),
        'num_buckets': 100,
        'hash_key': 1337,
        'name': 'single_ragged_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Dense Tensor of int32
    dense1 = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)
    input_dict_2 = {
        'inputs': np.array([dense1], dtype=object),
        'num_buckets': 50,
        'hash_key': None,
        'name': 'single_dense_int32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: RaggedTensor with num_buckets = 0 (no bucketing)
    rt2 = tf.ragged.constant([[100, 200], [300]], dtype=tf.int32)
    input_dict_3 = {
        'inputs': np.array([rt2], dtype=object),
        'num_buckets': 0,
        'hash_key': 123,
        'name': 'single_ragged_no_bucketing'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: RaggedTensor with empty rows
    rt3 = tf.ragged.constant([[10, 20], [], [30]], dtype=tf.int32)
    input_dict_4 = {
        'inputs': np.array([rt3], dtype=object),
        'num_buckets': 25,
        'hash_key': None,
        'name': 'single_ragged_with_empty_row'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: RaggedTensor that is entirely empty
    rt4 = tf.ragged.constant([[], []], dtype=tf.int64)
    input_dict_5 = {
        'inputs': np.array([rt4], dtype=object),
        'num_buckets': 10,
        'hash_key': 1,
        'name': 'single_ragged_all_empty'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Dense Tensor that is empty
    dense2 = tf.constant(np.zeros((2, 0)), dtype=tf.int32)
    input_dict_6 = {
        'inputs': np.array([dense2], dtype=object),
        'num_buckets': 5,
        'hash_key': None,
        'name': 'single_dense_empty'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: RaggedTensor with float32 dtype
    rt5 = tf.ragged.constant([[1.1, 2.2], [3.3]], dtype=tf.float32)
    input_dict_7 = {
        'inputs': np.array([rt5], dtype=object),
        'num_buckets': 150,
        'hash_key': 2024,
        'name': 'single_ragged_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: RaggedTensor with negative values
    rt6 = tf.ragged.constant([[-10, -20], [-30]], dtype=tf.int64)
    input_dict_8 = {
        'inputs': np.array([rt6], dtype=object),
        'num_buckets': 40,
        'hash_key': -50,
        'name': 'single_ragged_negative_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Dense Tensor with float32 dtype
    dense3 = tf.constant([[10.1, 10.2], [20.1, 20.2]], dtype=tf.float32)
    input_dict_9 = {
        'inputs': np.array([dense3], dtype=object),
        'num_buckets': 80,
        'hash_key': None,
        'name': 'single_dense_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Larger RaggedTensor with a specific hash_key
    rt7 = tf.ragged.constant([[1, 2, 3], [4], [5, 6], [7, 8, 9, 10]], dtype=tf.int64)
    input_dict_10 = {
        'inputs': np.array([rt7], dtype=object),
        'num_buckets': 5000,
        'hash_key': 888,
        'name': 'single_large_ragged'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.ragged.cross_hashed"] = get_tf_ragged_cross_hashed_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.ragged.cross_hashed' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ragged.cross_hashed'.")

check_valid('tf.ragged.cross_hashed', generated_inputs['tf.ragged.cross_hashed'], lib="tf", suffix=0)
