
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_sparse_reorder_inputs():
    """
    Generates a list of valid inputs for the tf.sparse.reorder function.
    The input 'sp_input' must be a tf.SparseTensor.
    """
    list_of_inputs = []

    # Input 1: Basic 2D case from the documentation
    indices = np.array([[0, 3], [0, 1], [3, 1], [2, 0]], dtype=np.int64)
    values = np.array([10, 20, 40, 30], dtype=np.int32)
    dense_shape = np.array([4, 5], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    input_dict = {
        'sp_input': sp_input,
        'name': 'reorder_2d_basic'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D tensor with out-of-order indices
    indices = np.array([[2, 1, 0], [0, 0, 1], [0, 1, 0]], dtype=np.int64)
    values = np.array([3.0, 1.0, 2.0], dtype=np.float32)
    dense_shape = np.array([3, 2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    input_dict = {
        'sp_input': sp_input,
        'name': 'reorder_3d_float'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Already ordered tensor (should be a no-op)
    indices = np.array([[0, 1], [0, 3], [2, 0], [3, 1]], dtype=np.int64)
    values = np.array([1, 2, 3, 4], dtype=np.int32)
    dense_shape = np.array([4, 5], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    input_dict = {
        'sp_input': sp_input,
        'name': 'already_ordered'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty sparse tensor
    indices = np.empty((0, 2), dtype=np.int64)
    values = np.array([], dtype=np.int32)
    dense_shape = np.array([10, 10], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    input_dict = {
        'sp_input': sp_input,
        'name': 'empty_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D sparse tensor (vector)
    indices = np.array([[8], [2], [5], [0]], dtype=np.int64)
    values = np.array([4, 2, 3, 1], dtype=np.int64)
    dense_shape = np.array([10], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    input_dict = {
        'sp_input': sp_input,
        'name': 'reorder_1d_vector'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Sparse tensor with negative values
    indices = np.array([[3, 0], [1, 1], [0, 2]], dtype=np.int64)
    values = np.array([-30, -20, -10], dtype=np.int32)
    dense_shape = np.array([4, 4], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    input_dict = {
        'sp_input': sp_input,
        'name': 'negative_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Sparse tensor with duplicate indices
    indices = np.array([[1, 1], [0, 0], [1, 1]], dtype=np.int64)
    values = np.array([2, 1, 3], dtype=np.int32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    input_dict = {
        'sp_input': sp_input,
        'name': 'duplicate_indices'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Higher rank tensor (4D)
    indices = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.int64)
    values = np.array([2.5, 1.5, 0.5], dtype=np.float64)
    dense_shape = np.array([1, 1, 2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    input_dict = {
        'sp_input': sp_input,
        'name': 'reorder_4d_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: A tensor with a large shape but few elements
    indices = np.array([[999, 999], [0, 0], [500, 2]], dtype=np.int64)
    values = np.array([3, 1, 2], dtype=np.int32)
    dense_shape = np.array([1000, 1000], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    input_dict = {
        'sp_input': sp_input,
        'name': 'large_sparse_shape'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Complex values
    indices = np.array([[1, 0], [0, 1]], dtype=np.int64)
    values = np.array([2+3j, 1+2j], dtype=np.complex64)
    dense_shape = np.array([2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    input_dict = {
        'sp_input': sp_input,
        'name': 'complex_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: All indices in the same first dimension with boolean values
    indices = np.array([[1, 4], [1, 0], [1, 2]], dtype=np.int64)
    values = np.array([True, False, True], dtype=bool)
    dense_shape = np.array([2, 5], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    input_dict = {
        'sp_input': sp_input,
        'name': 'same_first_dim_bool'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sparse.reorder"] = get_tf_sparse_reorder_inputs()

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
