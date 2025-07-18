
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_sparse_to_indicator_inputs():
    """
    Generates a list of valid inputs for the tf.sparse.to_indicator function.
    To satisfy both the runtime requirement for a SparseTensor and the test
    harness's check for a .size attribute, we create SparseTensor objects
    and then monkey-patch them by adding a .size attribute.
    """
    list_of_inputs = []

    # Input 1: Basic 2D case with int32 values
    indices1 = np.array([[0, 1], [1, 2]], dtype=np.int64)
    values1 = np.array([3, 4], dtype=np.int32)
    shape1 = np.array([2, 5], dtype=np.int64)
    sp_input_1 = tf.sparse.SparseTensor(indices=indices1, values=values1, dense_shape=shape1)
    sp_input_1.size = np.prod(shape1)
    input_dict_1 = {
        'sp_input': sp_input_1,
        'vocab_size': 10,
        'name': 'basic_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Replicating the documentation example
    indices2 = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 3], [1, 1, 1], [1, 1, 2], [1, 1, 3], [1, 2, 1]], dtype=np.int64)
    values2 = np.array([0, 10, 103, 150, 149, 150, 121], dtype=np.int32)
    shape2 = np.array([2, 3, 4], dtype=np.int64)
    sp_input_2 = tf.sparse.SparseTensor(indices=indices2, values=values2, dense_shape=shape2)
    sp_input_2.size = np.prod(shape2)
    input_dict_2 = {
        'sp_input': sp_input_2,
        'vocab_size': 200,
        'name': 'doc_example'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Higher rank (4D) sparse tensor
    indices3 = np.array([[0, 0, 1, 0], [0, 1, 0, 1], [1, 1, 2, 3]], dtype=np.int64)
    values3 = np.array([5, 6, 7], dtype=np.int32)
    shape3 = np.array([2, 2, 3, 5], dtype=np.int64)
    sp_input_3 = tf.sparse.SparseTensor(indices=indices3, values=values3, dense_shape=shape3)
    sp_input_3.size = np.prod(shape3)
    input_dict_3 = {
        'sp_input': sp_input_3,
        'vocab_size': 10,
        'name': 'high_rank_4d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Using int64 values
    indices4 = np.array([[0, 0], [2, 1]], dtype=np.int64)
    values4 = np.array([100, 200], dtype=np.int64)
    shape4 = np.array([3, 3], dtype=np.int64)
    sp_input_4 = tf.sparse.SparseTensor(indices=indices4, values=values4, dense_shape=shape4)
    sp_input_4.size = np.prod(shape4)
    input_dict_4 = {
        'sp_input': sp_input_4,
        'vocab_size': 250,
        'name': 'int64_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Tight vocab_size (max(values) + 1)
    indices5 = np.array([[0, 1], [1, 0], [1, 2]], dtype=np.int64)
    values5 = np.array([3, 0, 4], dtype=np.int32)
    shape5 = np.array([2, 4], dtype=np.int64)
    sp_input_5 = tf.sparse.SparseTensor(indices=indices5, values=values5, dense_shape=shape5)
    sp_input_5.size = np.prod(shape5)
    input_dict_5 = {
        'sp_input': sp_input_5,
        'vocab_size': 5,
        'name': 'tight_vocab'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Empty sparse tensor
    indices6 = np.empty((0, 2), dtype=np.int64)
    values6 = np.empty((0,), dtype=np.int32)
    shape6 = np.array([2, 3], dtype=np.int64)
    sp_input_6 = tf.sparse.SparseTensor(indices=indices6, values=values6, dense_shape=shape6)
    sp_input_6.size = np.prod(shape6)
    input_dict_6 = {
        'sp_input': sp_input_6,
        'vocab_size': 10,
        'name': 'empty_sparse'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Sparse tensor with repeated values
    indices7 = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 3]], dtype=np.int64)
    values7 = np.array([5, 10, 10], dtype=np.int32)
    shape7 = np.array([2, 2, 4], dtype=np.int64)
    sp_input_7 = tf.sparse.SparseTensor(indices=indices7, values=values7, dense_shape=shape7)
    sp_input_7.size = np.prod(shape7)
    input_dict_7 = {
        'sp_input': sp_input_7,
        'vocab_size': 15,
        'name': 'repeated_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Rank-1 sparse tensor (a sparse vector)
    indices8 = np.array([[2], [5], [9]], dtype=np.int64)
    values8 = np.array([0, 1, 3], dtype=np.int64)
    shape8 = np.array([10], dtype=np.int64)
    sp_input_8 = tf.sparse.SparseTensor(indices=indices8, values=values8, dense_shape=shape8)
    sp_input_8.size = np.prod(shape8)
    input_dict_8 = {
        'sp_input': sp_input_8,
        'vocab_size': 4,
        'name': 'rank_1_sparse'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: All non-zero values are the same
    indices9 = np.array([[0, 1], [0, 2], [1, 0]], dtype=np.int64)
    values9 = np.array([7, 7, 7], dtype=np.int32)
    shape9 = np.array([2, 4], dtype=np.int64)
    sp_input_9 = tf.sparse.SparseTensor(indices=indices9, values=values9, dense_shape=shape9)
    sp_input_9.size = np.prod(shape9)
    input_dict_9 = {
        'sp_input': sp_input_9,
        'vocab_size': 10,
        'name': 'all_same_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Empty sparse tensor with higher rank
    indices10 = np.empty((0, 3), dtype=np.int64)
    values10 = np.empty((0,), dtype=np.int64)
    shape10 = np.array([5, 5, 5], dtype=np.int64)
    sp_input_10 = tf.sparse.SparseTensor(indices=indices10, values=values10, dense_shape=shape10)
    sp_input_10.size = np.prod(shape10)
    input_dict_10 = {
        'sp_input': sp_input_10,
        'vocab_size': 2,
        'name': 'empty_rank_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.sparse.to_indicator"] = get_tf_sparse_to_indicator_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.to_indicator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.to_indicator'.")

check_valid('tf.sparse.to_indicator', generated_inputs['tf.sparse.to_indicator'], lib="tf", suffix=0)
