
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def get_tf_sparse_expand_dims_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D tensor, expand at axis=0 (add batch dimension)
    input_dict_1 = {
        'sp_input': tf.sparse.SparseTensor(
            indices=np.array([[0, 1], [1, 0]], dtype=np.int64),
            values=np.array([1, 2], dtype=np.int32),
            dense_shape=np.array([2, 2], dtype=np.int64)
        ),
        'axis': 0,
        'name': 'expand_2d_at_0'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic 2D tensor, expand at axis=1
    input_dict_2 = {
        'sp_input': tf.sparse.SparseTensor(
            indices=np.array([[0, 1], [1, 0]], dtype=np.int64),
            values=np.array([1, 2], dtype=np.int32),
            dense_shape=np.array([2, 2], dtype=np.int64)
        ),
        'axis': 1,
        'name': 'expand_2d_at_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Basic 2D tensor, expand at axis=-1 (add inner dimension)
    input_dict_3 = {
        'sp_input': tf.sparse.SparseTensor(
            indices=np.array([[0, 1], [1, 0]], dtype=np.int64),
            values=np.array([1, 2], dtype=np.int32),
            dense_shape=np.array([2, 2], dtype=np.int64)
        ),
        'axis': -1,
        'name': 'expand_2d_at_neg_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 3D tensor from docs, expand at axis=0
    input_dict_4 = {
        'sp_input': tf.sparse.SparseTensor(
            indices=np.array([[3, 4, 1]], dtype=np.int64),
            values=np.array([7], dtype=np.int32),
            dense_shape=np.array([10, 10, 3], dtype=np.int64)
        ),
        'axis': 0,
        'name': 'expand_3d_at_0'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 3D tensor, expand at axis=2
    input_dict_5 = {
        'sp_input': tf.sparse.SparseTensor(
            indices=np.array([[3, 4, 1]], dtype=np.int64),
            values=np.array([7], dtype=np.int32),
            dense_shape=np.array([10, 10, 3], dtype=np.int64)
        ),
        'axis': 2,
        'name': 'expand_3d_at_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 3D tensor, expand at axis=-2
    input_dict_6 = {
        'sp_input': tf.sparse.SparseTensor(
            indices=np.array([[3, 4, 1]], dtype=np.int64),
            values=np.array([7], dtype=np.int32),
            dense_shape=np.array([10, 10, 3], dtype=np.int64)
        ),
        'axis': -2,
        'name': 'expand_3d_at_neg_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 1D tensor, expand at axis=0
    input_dict_7 = {
        'sp_input': tf.sparse.SparseTensor(
            indices=np.array([[1], [4]], dtype=np.int64),
            values=np.array([10.0, 20.0], dtype=np.float32),
            dense_shape=np.array([5], dtype=np.int64)
        ),
        'axis': 0,
        'name': 'expand_1d_at_0'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 1D tensor, expand at axis=1 (max positive axis)
    input_dict_8 = {
        'sp_input': tf.sparse.SparseTensor(
            indices=np.array([[1], [4]], dtype=np.int64),
            values=np.array([10, 20], dtype=np.int32),
            dense_shape=np.array([5], dtype=np.int64)
        ),
        'axis': 1,
        'name': 'expand_1d_at_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Empty sparse tensor
    input_dict_9 = {
        'sp_input': tf.sparse.SparseTensor(
            indices=np.empty((0, 2), dtype=np.int64),
            values=np.array([], dtype=np.int32),
            dense_shape=np.array([5, 6], dtype=np.int64)
        ),
        'axis': 1,
        'name': 'expand_empty'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: 4D tensor, expand at axis=4 (max positive axis)
    input_dict_10 = {
        'sp_input': tf.sparse.SparseTensor(
            indices=np.array([[0, 0, 0, 0]], dtype=np.int64),
            values=np.array([1], dtype=np.int64),
            dense_shape=np.array([1, 1, 1, 1], dtype=np.int64)
        ),
        'axis': 4,
        'name': 'expand_4d_at_max_pos'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: High-rank tensor
    input_dict_11 = {
        'sp_input': tf.sparse.SparseTensor(
            indices=np.array([[0,1,2,3,4]], dtype=np.int64),
            values=np.array([100], dtype=np.int32),
            dense_shape=np.array([2,3,4,5,6], dtype=np.int64)
        ),
        'axis': -3,
        'name': 'expand_high_rank'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.sparse.expand_dims"] = get_tf_sparse_expand_dims_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.expand_dims' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.expand_dims'.")

check_valid('tf.sparse.expand_dims', generated_inputs['tf.sparse.expand_dims'], lib="tf", suffix=0)
