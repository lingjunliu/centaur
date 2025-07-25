
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_reorder_inputs():
    """
    Generates a list of valid inputs for the tf.sparse.reorder function.
    """
    list_of_inputs = []

    # Input 1: Basic case from the documentation (using integers)
    sp_input_1 = tf.SparseTensor(
        indices=np.array([[0, 3], [0, 1], [3, 1], [2, 0]], dtype=np.int64),
        values=np.array([2, 1, 4, 3], dtype=np.int32),
        dense_shape=np.array([4, 5], dtype=np.int64)
    )
    input_dict_1 = {'sp_input': sp_input_1, 'name': 'doc_example_int'}
    list_of_inputs.append(input_dict_1)

    # Input 2: Already ordered input (int32 values)
    sp_input_2 = tf.SparseTensor(
        indices=np.array([[0, 1], [0, 3], [2, 0], [3, 1]], dtype=np.int64),
        values=np.array([10, 20, 30, 40], dtype=np.int32),
        dense_shape=np.array([4, 5], dtype=np.int64)
    )
    input_dict_2 = {'sp_input': sp_input_2, 'name': 'already_ordered'}
    list_of_inputs.append(input_dict_2)

    # Input 3: 3D sparse tensor (float32 values)
    sp_input_3 = tf.SparseTensor(
        indices=np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=np.int64),
        values=np.array([3.3, 2.2, 4.4, 1.1], dtype=np.float32),
        dense_shape=np.array([2, 2, 2], dtype=np.int64)
    )
    input_dict_3 = {'sp_input': sp_input_3, 'name': '3d_float_tensor'}
    list_of_inputs.append(input_dict_3)

    # Input 4: Empty sparse tensor (no non-zero elements)
    sp_input_4 = tf.SparseTensor(
        indices=np.empty((0, 3), dtype=np.int64),
        values=np.array([], dtype=np.float32),
        dense_shape=np.array([5, 5, 5], dtype=np.int64)
    )
    input_dict_4 = {'sp_input': sp_input_4, 'name': 'empty_tensor'}
    list_of_inputs.append(input_dict_4)

    # Input 5: 1D sparse tensor (vector) with negative values
    sp_input_5 = tf.SparseTensor(
        indices=np.array([[8], [2], [5], [0]], dtype=np.int64),
        values=np.array([-10, -2, -5, -1], dtype=np.int64),
        dense_shape=np.array([10], dtype=np.int64)
    )
    input_dict_5 = {'sp_input': sp_input_5, 'name': None}
    list_of_inputs.append(input_dict_5)

    # Input 6: 4D sparse tensor with integer values
    sp_input_6 = tf.SparseTensor(
        indices=np.array([[0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.int64),
        values=np.array([1, 0], dtype=np.int32),
        dense_shape=np.array([1, 1, 1, 2], dtype=np.int64)
    )
    input_dict_6 = {'sp_input': sp_input_6, 'name': '4d_int_tensor'}
    list_of_inputs.append(input_dict_6)

    # Input 7: Sparse tensor with duplicate indices
    sp_input_7 = tf.SparseTensor(
        indices=np.array([[1, 1], [0, 0], [1, 1]], dtype=np.int64),
        values=np.array([10, 20, 30], dtype=np.int32),
        dense_shape=np.array([2, 2], dtype=np.int64)
    )
    input_dict_7 = {'sp_input': sp_input_7, 'name': 'duplicate_indices'}
    list_of_inputs.append(input_dict_7)

    # Input 8: Single element sparse tensor with float64 values
    sp_input_8 = tf.SparseTensor(
        indices=np.array([[3, 4]], dtype=np.int64),
        values=np.array([100.5], dtype=np.float64),
        dense_shape=np.array([5, 5], dtype=np.int64)
    )
    input_dict_8 = {'sp_input': sp_input_8, 'name': 'single_element'}
    list_of_inputs.append(input_dict_8)
    
    # Input 9: All elements in the same row, unordered
    sp_input_9 = tf.SparseTensor(
        indices=np.array([[1, 4], [1, 0], [1, 9], [1, 2]], dtype=np.int64),
        values=np.array([4, 0, 9, 2], dtype=np.int32),
        dense_shape=np.array([2, 10], dtype=np.int64)
    )
    input_dict_9 = {'sp_input': sp_input_9, 'name': 'single_row_unordered'}
    list_of_inputs.append(input_dict_9)

    # Input 10: Reverse-ordered sparse tensor
    sp_input_10 = tf.SparseTensor(
        indices=np.array([[3, 3], [2, 2], [1, 1], [0, 0]], dtype=np.int64),
        values=np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32),
        dense_shape=np.array([4, 4], dtype=np.int64)
    )
    input_dict_10 = {'sp_input': sp_input_10, 'name': 'reverse_ordered'}
    list_of_inputs.append(input_dict_10)

    return list_of_inputs

generated_inputs["tf.sparse.reorder"] = tf_sparse_reorder_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.reorder' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.reorder'.")

check_valid('tf.sparse.reorder', generated_inputs['tf.sparse.reorder'], lib="tf", suffix=0)
