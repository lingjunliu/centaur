
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_sparse_cross_hashed_inputs():
    """
    Generates a list of valid inputs for the tf.sparse.cross_hashed function.
    The inputs are restricted to dense tensors of numerical types that can be stacked,
    due to limitations of the testing framework.
    """
    list_of_inputs = []

    # Input 1: Basic case with two 2D int32 tensors
    t1_1 = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)
    t1_2 = tf.constant([[5, 6], [7, 8]], dtype=tf.int32)
    input_dict_1 = {
        'inputs': tf.stack([t1_1, t1_2]),
        'num_buckets': 1000,
        'hash_key': 1337,
        'name': "dense_2d_int32_cross"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Crossing three 2D int64 tensors, no bucketing
    t2_1 = tf.constant([[10, 20], [30, 40]], dtype=tf.int64)
    t2_2 = tf.constant([[50, 60], [70, 80]], dtype=tf.int64)
    t2_3 = tf.constant([[90, 100], [110, 120]], dtype=tf.int64)
    input_dict_2 = {
        'inputs': tf.stack([t2_1, t2_2, t2_3]),
        'num_buckets': 0,
        'hash_key': 2024,
        'name': "dense_2d_int64_cross"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: "Crossing" a single float32 tensor (effectively just hashing)
    t3_1 = tf.constant([[1.1], [2.2], [3.3]], dtype=tf.float32)
    input_dict_3 = {
        'inputs': tf.stack([t3_1]),
        'num_buckets': 500,
        'hash_key': 98765,
        'name': "single_dense_float32_hash"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Using a single bucket with int16 tensors
    t4_1 = tf.constant([[1], [2]], dtype=tf.int16)
    t4_2 = tf.constant([[3], [4]], dtype=tf.int16)
    input_dict_4 = {
        'inputs': tf.stack([t4_1, t4_2]),
        'num_buckets': 1,
        'hash_key': 1,
        'name': "single_bucket_dense_cross"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Using a large hash key
    t5_1 = tf.constant([[101]], dtype=tf.int64)
    t5_2 = tf.constant([[202]], dtype=tf.int64)
    input_dict_5 = {
        'inputs': tf.stack([t5_1, t5_2]),
        'num_buckets': 10,
        'hash_key': 9223372036854775807,
        'name': "large_hash_key_cross"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Using a negative hash key
    t6_1 = tf.constant([[10, 20, 30]], dtype=tf.int32)
    t6_2 = tf.constant([[40, 50, 60]], dtype=tf.int32)
    input_dict_6 = {
        'inputs': tf.stack([t6_1, t6_2]),
        'num_buckets': 0,
        'hash_key': -1234567,
        'name': "negative_hash_key_cross"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Crossing two 3D float32 tensors
    t7_1 = tf.constant([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=tf.float32)
    t7_2 = tf.constant([[[5.0], [6.0]], [[7.0], [8.0]]], dtype=tf.float32)
    input_dict_7 = {
        'inputs': tf.stack([t7_1, t7_2]),
        'num_buckets': 200,
        'hash_key': 101112,
        'name': "dense_3d_float32_cross"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Empty input tensor list
    input_dict_8 = {
        'inputs': tf.constant([], shape=(0, 2, 2), dtype=tf.float32),
        'num_buckets': 256,
        'hash_key': 99,
        'name': "empty_input_list"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Crossing tensors with an empty batch dimension
    t9_1 = tf.constant([], shape=(0, 3), dtype=tf.int32)
    t9_2 = tf.constant([], shape=(0, 3), dtype=tf.int32)
    input_dict_9 = {
        'inputs': tf.stack([t9_1, t9_2]),
        'num_buckets': 150,
        'hash_key': 123,
        'name': "empty_batch_cross"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Crossing four tensors
    t10_1 = tf.constant([[1], [2]], dtype=tf.int16)
    t10_2 = tf.constant([[3], [4]], dtype=tf.int16)
    t10_3 = tf.constant([[5], [6]], dtype=tf.int16)
    t10_4 = tf.constant([[7], [8]], dtype=tf.int16)
    input_dict_10 = {
        'inputs': tf.stack([t10_1, t10_2, t10_3, t10_4]),
        'num_buckets': 100,
        'hash_key': 42,
        'name': "four_tensor_cross"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.sparse.cross_hashed"] = tf_sparse_cross_hashed_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.cross_hashed' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.cross_hashed'.")

check_valid('tf.sparse.cross_hashed', generated_inputs['tf.sparse.cross_hashed'], lib="tf", suffix=0)
