
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_sparse_dense_matmul_inputs():
    list_of_inputs = []

    # Input 1
    sp_a = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[2, 3])
    b = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    adjoint_a = False
    adjoint_b = False
    name = "test1"

    input_dict = {
        "sp_a": sp_a,
        "b": b,
        "adjoint_a": adjoint_a,
        "adjoint_b": adjoint_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    sp_a = tf.sparse.SparseTensor(indices=[[0, 1], [1, 0]], values=[3, 4], dense_shape=[2, 2])
    b = np.array([[5, 6], [7, 8]], dtype=np.float32)
    adjoint_a = True
    adjoint_b = True
    name = "test2"

    input_dict = {
        "sp_a": sp_a,
        "b": b,
        "adjoint_a": adjoint_a,
        "adjoint_b": adjoint_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    sp_a = tf.sparse.SparseTensor(indices=[[0, 0], [0, 1], [1, 1], [1, 2]], values=[1, 2, 3, 4], dense_shape=[2, 3])
    b = np.array([[1], [2], [3]], dtype=np.float32)
    adjoint_a = False
    adjoint_b = True
    name = "test3"

    input_dict = {
        "sp_a": sp_a,
        "b": b,
        "adjoint_a": adjoint_a,
        "adjoint_b": adjoint_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    sp_a = tf.sparse.SparseTensor(indices=[[0, 0]], values=[5], dense_shape=[1, 1])
    b = np.array([[10]], dtype=np.float32)
    adjoint_a = True
    adjoint_b = False
    name = "test4"

    input_dict = {
        "sp_a": sp_a,
        "b": b,
        "adjoint_a": adjoint_a,
        "adjoint_b": adjoint_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    sp_a = tf.sparse.SparseTensor(indices=[[0, 0], [1, 1], [2, 2]], values=[1, 1, 1], dense_shape=[3, 3])
    b = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    adjoint_a = False
    adjoint_b = False
    name = "test5"

    input_dict = {
        "sp_a": sp_a,
        "b": b,
        "adjoint_a": adjoint_a,
        "adjoint_b": adjoint_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    sp_a = tf.sparse.SparseTensor(indices=[[0, 0], [0, 1]], values=[1.5, 2.5], dense_shape=[1, 2])
    b = np.array([[3.5], [4.5]], dtype=np.float32)
    adjoint_a = False
    adjoint_b = False
    name = "test6"

    input_dict = {
        "sp_a": sp_a,
        "b": b,
        "adjoint_a": adjoint_a,
        "adjoint_b": adjoint_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    sp_a = tf.sparse.SparseTensor(indices=[[0, 0], [1, 0], [1, 1]], values=[-1, -2, -3], dense_shape=[2, 2])
    b = np.array([[1, 2], [3, 4]], dtype=np.float32)
    adjoint_a = False
    adjoint_b = False
    name = "test7"

    input_dict = {
        "sp_a": sp_a,
        "b": b,
        "adjoint_a": adjoint_a,
        "adjoint_b": adjoint_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    sp_a = tf.sparse.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [1, 1]], values=[1, 2, 3, 4], dense_shape=[2, 2])
    b = np.array([[5, 6], [7, 8]], dtype=np.float32)
    adjoint_a = True
    adjoint_b = False
    name = "test8"

    input_dict = {
        "sp_a": sp_a,
        "b": b,
        "adjoint_a": adjoint_a,
        "adjoint_b": adjoint_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    sp_a = tf.sparse.SparseTensor(indices=[[0, 0]], values=[1], dense_shape=[1, 1])
    b = np.array([[2,3,4]], dtype=np.float32)
    adjoint_a = False
    adjoint_b = True
    name = "test9"

    input_dict = {
        "sp_a": sp_a,
        "b": b,
        "adjoint_a": adjoint_a,
        "adjoint_b": adjoint_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    sp_a = tf.sparse.SparseTensor(indices=[[0, 0], [1, 1]], values=[1, 1], dense_shape=[2, 2])
    b = np.array([[0, 1], [1, 0]], dtype=np.float32)
    adjoint_a = True
    adjoint_b = True
    name = "test10"

    input_dict = {
        "sp_a": sp_a,
        "b": b,
        "adjoint_a": adjoint_a,
        "adjoint_b": adjoint_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.sparse_dense_matmul"] = tf_sparse_sparse_dense_matmul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.sparse_dense_matmul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.sparse_dense_matmul'.")

check_valid('tf.sparse.sparse_dense_matmul', generated_inputs['tf.sparse.sparse_dense_matmul'], lib="tf", suffix=0)
