
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_sparse_dense_matmul_inputs():
    list_of_inputs = []

    # Case 1: sp_a is sparse, b is dense. dtype=float32.
    sp_a_1 = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1.0, 2.0], dense_shape=[2, 3])
    b_1 = np.array([[10., 11.], [20., 21.], [30., 31.]], dtype=np.float32)
    input_dict_1 = {'sp_a': sp_a_1, 'b': b_1, 'adjoint_a': False, 'adjoint_b': False, 'name': 'case1'}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: sp_a is dense, b is sparse. dtype=float32.
    sp_a_2 = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    b_2 = tf.SparseTensor(indices=[[0, 1], [2, 0]], values=[10.0, 20.0], dense_shape=[3, 2])
    input_dict_2 = {'sp_a': sp_a_2, 'b': b_2, 'adjoint_a': False, 'adjoint_b': False, 'name': 'case2'}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: sp_a is sparse, b is dense. dtype=int32.
    sp_a_3 = tf.SparseTensor(indices=[[0, 1], [2, 1]], values=[3, 4], dense_shape=[3, 2])
    b_3 = np.array([[10, 11, 12], [20, 21, 22]], dtype=np.int32)
    input_dict_3 = {'sp_a': sp_a_3, 'b': b_3, 'adjoint_a': False, 'adjoint_b': False, 'name': 'case3'}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: sp_a is dense, b is sparse. dtype=int32.
    sp_a_4 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    b_4 = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[10, 20], dense_shape=[2, 3])
    input_dict_4 = {'sp_a': sp_a_4, 'b': b_4, 'adjoint_a': False, 'adjoint_b': False, 'name': 'case4'}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: With adjoint_a=True. sp_a [3,2] becomes [2,3] and multiplies b [3,1].
    sp_a_5 = tf.SparseTensor(indices=[[0, 1], [2, 0]], values=[1.0, 2.0], dense_shape=[3, 2])
    b_5 = np.array([[10.], [20.], [30.]], dtype=np.float32)
    input_dict_5 = {'sp_a': sp_a_5, 'b': b_5, 'adjoint_a': True, 'adjoint_b': False, 'name': 'case5'}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: With adjoint_b=True. sp_a [2,3] multiplies b [4,3] which becomes [3,4].
    sp_a_6 = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1.0, 2.0], dense_shape=[2, 3])
    b_6 = np.arange(12, dtype=np.float32).reshape(4, 3)
    input_dict_6 = {'sp_a': sp_a_6, 'b': b_6, 'adjoint_a': False, 'adjoint_b': True, 'name': 'case6'}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: Complex numbers, sp_a is sparse.
    sp_a_7 = tf.SparseTensor(indices=[[0,1]], values=[(1+2j)], dense_shape=[2,2])
    b_7 = np.array([[(1+1j), (2+2j)], [(3+3j), (4+4j)]], dtype=np.complex64)
    input_dict_7 = {'sp_a': sp_a_7, 'b': b_7, 'adjoint_a': False, 'adjoint_b': False, 'name': 'case7'}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: Complex numbers, b is sparse.
    sp_a_8 = np.array([[(1+1j), (2+2j)], [(3+3j), (4+4j)]], dtype=np.complex64)
    b_8 = tf.SparseTensor(indices=[[1,0]], values=[(5-5j)], dense_shape=[2,1])
    input_dict_8 = {'sp_a': sp_a_8, 'b': b_8, 'adjoint_a': False, 'adjoint_b': False, 'name': 'case8'}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: Empty sparse tensor
    sp_a_9 = tf.SparseTensor(indices=np.empty((0, 2), dtype=np.int64), values=[], dense_shape=[4, 5])
    b_9 = np.random.rand(5, 6).astype(np.float32)
    input_dict_9 = {'sp_a': sp_a_9, 'b': b_9, 'adjoint_a': False, 'adjoint_b': False, 'name': 'case9'}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: Matrix-vector multiplication
    sp_a_10 = tf.SparseTensor(indices=[[0,1], [3,0]], values=[1., -1.], dense_shape=[4, 2])
    b_10 = np.array([[100.], [200.]], dtype=np.float32)
    input_dict_10 = {'sp_a': sp_a_10, 'b': b_10, 'adjoint_a': False, 'adjoint_b': False, 'name': 'case10'}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.sparse.sparse_dense_matmul"] = get_sparse_dense_matmul_inputs()

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
