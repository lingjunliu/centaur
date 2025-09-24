
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_sparse_reshape_inputs():
    """
    Generates a list of valid inputs for the tf.sparse.reshape function.
    """
    list_of_inputs = []

    # Input 1: Basic 2D to 2D reshape
    sp_input_1 = tf.SparseTensor(
        indices=np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64),
        values=np.array([10, 20, 30], dtype=np.int32),
        dense_shape=np.array([3, 4], dtype=np.int64)
    )
    shape_1 = np.array([2, 6], dtype=np.int64)
    input_dict_1 = {
        'sp_input': sp_input_1,
        'shape': shape_1,
        'name': 'basic_2d_reshape'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Flatten a 3D tensor to 1D
    sp_input_2 = tf.SparseTensor(
        indices=np.array([[0, 1, 1], [1, 0, 2]], dtype=np.int64),
        values=np.array([1.1, 2.2], dtype=np.float32),
        dense_shape=np.array([2, 2, 3], dtype=np.int64)
    )
    shape_2 = np.array([12], dtype=np.int64)
    input_dict_2 = {
        'sp_input': sp_input_2,
        'shape': shape_2,
        'name': 'flatten_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Reshape with an inferred dimension (-1)
    sp_input_3 = tf.SparseTensor(
        indices=np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 2, 3]], dtype=np.int64),
        values=np.array([1, 2, 3, 4, 5], dtype=np.int64),
        dense_shape=np.array([2, 3, 6], dtype=np.int64)
    )
    shape_3 = np.array([9, -1], dtype=np.int64)
    input_dict_3 = {
        'sp_input': sp_input_3,
        'shape': shape_3,
        'name': 'inferred_dim'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Unflatten a 1D tensor to 3D with string values
    sp_input_4 = tf.SparseTensor(
        indices=np.array([[1], [8], [15]], dtype=np.int64),
        values=np.array([b'x', b'y', b'z'], dtype=object), # Use bytes for string tensors
        dense_shape=np.array([18], dtype=np.int64)
    )
    shape_4 = np.array([2, 3, 3], dtype=np.int64)
    input_dict_4 = {
        'sp_input': sp_input_4,
        'shape': shape_4,
        'name': 'unflatten_1d_string'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Reshaping an empty sparse tensor
    sp_input_5 = tf.SparseTensor(
        indices=np.empty((0, 2), dtype=np.int64),
        values=np.array([], dtype=np.float64),
        dense_shape=np.array([5, 10], dtype=np.int64)
    )
    shape_5 = np.array([2, 25], dtype=np.int64)
    input_dict_5 = {
        'sp_input': sp_input_5,
        'shape': shape_5,
        'name': 'empty_reshape'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Reshaping a "full" sparse tensor
    sp_input_6 = tf.SparseTensor(
        indices=np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64),
        values=np.array([1, 2, 3, 4], dtype=np.int32),
        dense_shape=np.array([2, 2], dtype=np.int64)
    )
    shape_6 = np.array([4, 1], dtype=np.int64)
    input_dict_6 = {
        'sp_input': sp_input_6,
        'shape': shape_6,
        'name': 'full_reshape'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Reshape a higher-rank tensor (4D to 2D) with boolean values
    sp_input_7 = tf.SparseTensor(
        indices=np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int64),
        values=np.array([True, False], dtype=bool),
        dense_shape=np.array([2, 2, 2, 2], dtype=np.int64)
    )
    shape_7 = np.array([4, 4], dtype=np.int64)
    input_dict_7 = {
        'sp_input': sp_input_7,
        'shape': shape_7,
        'name': '4d_to_2d_bool'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Single element sparse tensor
    sp_input_8 = tf.SparseTensor(
        indices=np.array([[2, 3]], dtype=np.int64),
        values=np.array([99], dtype=np.int32),
        dense_shape=np.array([5, 5], dtype=np.int64)
    )
    shape_8 = np.array([25], dtype=np.int64)
    input_dict_8 = {
        'sp_input': sp_input_8,
        'shape': shape_8,
        'name': 'single_element_reshape'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: Complex numbers as values
    sp_input_9 = tf.SparseTensor(
        indices=np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int64),
        values=np.array([1+2j, 3+4j, 5+6j, 7+8j], dtype=np.complex64),
        dense_shape=np.array([4, 4], dtype=np.int64)
    )
    shape_9 = np.array([2, -1], dtype=np.int64)
    input_dict_9 = {
        'sp_input': sp_input_9,
        'shape': shape_9,
        'name': 'complex_reshape'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: Reshaping from 1xN to Nx1
    sp_input_10 = tf.SparseTensor(
        indices=np.array([[0, 3], [0, 7]], dtype=np.int64),
        values=np.array([10, 20], dtype=np.int32),
        dense_shape=np.array([1, 10], dtype=np.int64)
    )
    shape_10 = np.array([10, 1], dtype=np.int64)
    input_dict_10 = {
        'sp_input': sp_input_10,
        'shape': shape_10,
        'name': 'row_to_col_vector'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.sparse.reshape"] = get_tf_sparse_reshape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.reshape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.reshape'.")

check_valid('tf.sparse.reshape', generated_inputs['tf.sparse.reshape'], lib="tf", suffix=0)
