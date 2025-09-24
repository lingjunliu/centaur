
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_tf_sparse_expand_dims_inputs():
    list_of_inputs = []

    # The tf.sparse.expand_dims function requires a tf.SparseTensor as input.
    # The error "'EagerTensor' object has no attribute 'dense_shape'" indicates that
    # a dense tensor (which is not a valid input type) was provided.
    # The correct approach is to create tf.SparseTensor objects for the `sp_input` parameter.
    # The components of these SparseTensors (indices, values, dense_shape) are
    # created from numpy arrays to adhere to the "numpy format" instruction.

    # Input 1: Basic 2D tensor, expand at axis 0
    sp_input_1 = tf.SparseTensor(
        indices=np.array([[0, 1], [1, 2]], dtype=np.int64),
        values=np.array([10, 20], dtype=np.int32),
        dense_shape=np.array([3, 4], dtype=np.int64)
    )
    input_dict_1 = {'sp_input': sp_input_1, 'axis': 0, 'name': 'expand_axis_0'}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic 2D tensor, expand at axis 1, float values
    sp_input_2 = tf.SparseTensor(
        indices=np.array([[0, 1], [1, 2]], dtype=np.int64),
        values=np.array([10.0, 20.0], dtype=np.float32),
        dense_shape=np.array([3, 4], dtype=np.int64)
    )
    input_dict_2 = {'sp_input': sp_input_2, 'axis': 1, 'name': 'expand_axis_1'}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D tensor, expand at axis -1 (add inner dimension)
    sp_input_3 = tf.SparseTensor(
        indices=np.array([[0, 1], [1, 2]], dtype=np.int64),
        values=np.array([b'a', b'b'], dtype=object),
        dense_shape=np.array([3, 4], dtype=np.int64)
    )
    input_dict_3 = {'sp_input': sp_input_3, 'axis': -1, 'name': 'expand_axis_neg_1'}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 3D tensor, expand at axis 0
    sp_input_4 = tf.SparseTensor(
        indices=np.array([[0, 1, 0], [1, 2, 1]], dtype=np.int64),
        values=np.array([1, 2], dtype=np.int64),
        dense_shape=np.array([2, 3, 2], dtype=np.int64)
    )
    input_dict_4 = {'sp_input': sp_input_4, 'axis': 0, 'name': '3d_expand_0'}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 3D tensor, expand at axis -2, boolean values
    sp_input_5 = tf.SparseTensor(
        indices=np.array([[0, 1, 0], [1, 2, 1]], dtype=np.int64),
        values=np.array([True, False], dtype=bool),
        dense_shape=np.array([2, 3, 2], dtype=np.int64)
    )
    input_dict_5 = {'sp_input': sp_input_5, 'axis': -2, 'name': '3d_expand_neg_2'}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 3D tensor, expand at axis 3 (end of shape)
    sp_input_6 = tf.SparseTensor(
        indices=np.array([[0, 1, 0], [1, 2, 1]], dtype=np.int64),
        values=np.array([1, 2], dtype=np.int32),
        dense_shape=np.array([2, 3, 2], dtype=np.int64)
    )
    input_dict_6 = {'sp_input': sp_input_6, 'axis': 3, 'name': '3d_expand_end'}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 1D tensor, expand at axis 0
    sp_input_7 = tf.SparseTensor(
        indices=np.array([[1], [3]], dtype=np.int64),
        values=np.array([1.0, 3.0], dtype=np.float64),
        dense_shape=np.array([5], dtype=np.int64)
    )
    input_dict_7 = {'sp_input': sp_input_7, 'axis': 0, 'name': '1d_expand_0'}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 1D tensor, expand at axis -2 (start of shape)
    sp_input_8 = tf.SparseTensor(
        indices=np.array([[1], [3]], dtype=np.int64),
        values=np.array([1, 3], dtype=np.int16),
        dense_shape=np.array([5], dtype=np.int64)
    )
    input_dict_8 = {'sp_input': sp_input_8, 'axis': -2, 'name': '1d_expand_start'}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Empty sparse tensor (no non-zero elements), expand at axis 1
    sp_input_9 = tf.SparseTensor(
        indices=np.empty((0, 3), dtype=np.int64),
        values=np.empty((0,), dtype=np.int32),
        dense_shape=np.array([4, 5, 6], dtype=np.int64)
    )
    input_dict_9 = {'sp_input': sp_input_9, 'axis': 1, 'name': 'empty_expand'}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Higher rank tensor (4D), expand in middle
    sp_input_10 = tf.SparseTensor(
        indices=np.array([[0, 0, 0, 0]], dtype=np.int64),
        values=np.array([100], dtype=np.int32),
        dense_shape=np.array([2, 2, 2, 2], dtype=np.int64)
    )
    input_dict_10 = {'sp_input': sp_input_10, 'axis': 2, 'name': '4d_expand_2'}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Rank 5 tensor, expand at maximum negative axis
    sp_input_11 = tf.SparseTensor(
        indices=np.array([[1, 2, 3, 4, 5]], dtype=np.int64),
        values=np.array([99], dtype=np.int32),
        dense_shape=np.array([10, 10, 10, 10, 10], dtype=np.int64)
    )
    input_dict_11 = {'sp_input': sp_input_11, 'axis': -6, 'name': 'rank5_expand_begin'}
    list_of_inputs.append(copy.deepcopy(input_dict_11))
    
    return list_of_inputs

generated_inputs["tf.sparse.expand_dims"] = generate_tf_sparse_expand_dims_inputs()

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
