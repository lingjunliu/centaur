
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_sparse_dense_matmul_inputs():
    list_of_inputs = []

    # Input 1: sp_a is SparseTensor, b is dense Tensor, sorted indices
    indices1 = np.array([[0, 1], [1, 3], [2, 0]], dtype=np.int64)
    values1 = np.array([1.0, 2.0, -3.0], dtype=np.float32)
    dense_shape1 = np.array([3, 4], dtype=np.int64)
    sp_a1 = tf.sparse.reorder(tf.SparseTensor(indices=indices1, values=values1, dense_shape=dense_shape1))
    b1 = tf.constant(np.arange(8, dtype=np.float32).reshape(4, 2))
    input_dict_1 = {
        'sp_a': sp_a1,
        'b': b1,
        'adjoint_a': False,
        'adjoint_b': False,
        'name': 'sparse_a_dense_b_sorted'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: sp_a is dense Tensor, b is SparseTensor, sorted indices
    sp_a2 = tf.constant(np.arange(12, dtype=np.float64).reshape(3, 4))
    indices2 = np.array([[0, 1], [2, 0], [3, 1]], dtype=np.int64)
    values2 = np.array([1.0, -2.5, 3.0], dtype=np.float64)
    dense_shape2 = np.array([4, 2], dtype=np.int64)
    b2 = tf.sparse.reorder(tf.SparseTensor(indices=indices2, values=values2, dense_shape=dense_shape2))
    input_dict_2 = {
        'sp_a': sp_a2,
        'b': b2,
        'adjoint_a': False,
        'adjoint_b': False,
        'name': 'dense_a_sparse_b_sorted'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: sp_a is SparseTensor, adjoint_a=True, column-major sorted indices
    indices3 = np.array([[3, 0], [1, 1], [0, 2], [2, 2]], dtype=np.int64) # Sorted by col, then row
    values3 = np.array([4.0, 2.0, 1.0, 3.0], dtype=np.float32)
    dense_shape3 = np.array([4, 3], dtype=np.int64)
    sp_a3 = tf.SparseTensor(indices=indices3, values=values3, dense_shape=dense_shape3)
    b3 = tf.constant(np.arange(8, dtype=np.float32).reshape(4, 2))
    input_dict_3 = {
        'sp_a': sp_a3,
        'b': b3,
        'adjoint_a': True,
        'adjoint_b': False,
        'name': 'sparse_a_adjoint_a_col_sorted'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: sp_a is SparseTensor, adjoint_b=True
    indices4 = np.array([[0, 1], [1, 3], [2, 0]], dtype=np.int64)
    values4 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dense_shape4 = np.array([3, 4], dtype=np.int64)
    sp_a4 = tf.sparse.reorder(tf.SparseTensor(indices=indices4, values=values4, dense_shape=dense_shape4))
    b4 = tf.constant(np.arange(8, dtype=np.float32).reshape(2, 4))
    input_dict_4 = {
        'sp_a': sp_a4,
        'b': b4,
        'adjoint_a': False,
        'adjoint_b': True,
        'name': 'sparse_a_adjoint_b'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: sp_a is dense, b is SparseTensor, both adjoints true
    sp_a5 = tf.constant(np.arange(12, dtype=np.int32).reshape(4, 3))
    indices5 = np.array([[0, 1], [1, 0], [1, 2]], dtype=np.int64)
    values5 = np.array([5, 6, 7], dtype=np.int32)
    dense_shape5 = np.array([2, 4], dtype=np.int64)
    b5 = tf.sparse.reorder(tf.SparseTensor(indices=indices5, values=values5, dense_shape=dense_shape5))
    input_dict_5 = {
        'sp_a': sp_a5,
        'b': b5,
        'adjoint_a': True,
        'adjoint_b': True,
        'name': 'dense_a_sparse_b_adjoints'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Integer types
    indices6 = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.int64)
    values6 = np.array([-1, -2, 3, 4, -5], dtype=np.int32)
    dense_shape6 = np.array([5, 5], dtype=np.int64)
    sp_a6 = tf.SparseTensor(indices=indices6, values=values6, dense_shape=dense_shape6)
    b6 = tf.constant(np.arange(15, dtype=np.int32).reshape(5, 3))
    input_dict_6 = {
        'sp_a': sp_a6,
        'b': b6,
        'adjoint_a': False,
        'adjoint_b': False,
        'name': 'int32_types'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Complex numbers
    indices7 = np.array([[0, 1]], dtype=np.int64)
    values7 = np.array([1+2j], dtype=np.complex64)
    dense_shape7 = np.array([2, 2], dtype=np.int64)
    sp_a7 = tf.SparseTensor(indices=indices7, values=values7, dense_shape=dense_shape7)
    b7 = tf.constant(np.ones((2, 2), dtype=np.complex64))
    input_dict_7 = {
        'sp_a': sp_a7,
        'b': b7,
        'adjoint_a': False,
        'adjoint_b': False,
        'name': 'complex64_types'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Matrix-vector multiplication
    indices8 = np.array([[1, 5], [8, 19]], dtype=np.int64)
    values8 = np.array([10.5, -3.14], dtype=np.float64)
    dense_shape8 = np.array([10, 20], dtype=np.int64)
    sp_a8 = tf.sparse.reorder(tf.SparseTensor(indices=indices8, values=values8, dense_shape=dense_shape8))
    b8 = tf.constant(np.random.rand(20, 1).astype(np.float64))
    input_dict_8 = {
        'sp_a': sp_a8,
        'b': b8,
        'adjoint_a': False,
        'adjoint_b': False,
        'name': 'matrix_vector'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Empty sparse tensor
    indices9 = np.empty((0, 2), dtype=np.int64)
    values9 = np.empty(0, dtype=np.float32)
    dense_shape9 = np.array([3, 4], dtype=np.int64)
    sp_a9 = tf.SparseTensor(indices=indices9, values=values9, dense_shape=dense_shape9)
    b9 = tf.constant(np.ones((4, 2), dtype=np.float32))
    input_dict_9 = {
        'sp_a': sp_a9,
        'b': b9,
        'adjoint_a': False,
        'adjoint_b': False,
        'name': 'empty_sparse'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Empty sparse tensor for b
    sp_a10 = tf.constant(np.ones((3,4), dtype=np.float32))
    indices10 = np.empty((0, 2), dtype=np.int64)
    values10 = np.empty(0, dtype=np.float32)
    dense_shape10 = np.array([4, 2], dtype=np.int64)
    b10 = tf.SparseTensor(indices=indices10, values=values10, dense_shape=dense_shape10)
    input_dict_10 = {
        'sp_a': sp_a10,
        'b': b10,
        'adjoint_a': False,
        'adjoint_b': False,
        'name': 'empty_sparse_b'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.sparse.sparse_dense_matmul"] = tf_sparse_sparse_dense_matmul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.sparse_dense_matmul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.sparse_dense_matmul'.")

check_valid('tf.sparse.sparse_dense_matmul', generated_inputs['tf.sparse.sparse_dense_matmul'], lib="tf", suffix=0)
