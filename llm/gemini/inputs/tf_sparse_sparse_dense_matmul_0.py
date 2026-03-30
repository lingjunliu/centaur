
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_sparse_dense_matmul_inputs():
    list_of_inputs = []

    # Input 1
    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.float32)
    dense_shape = (2, 3)
    sp_a = tf.SparseTensor(indices, values, dense_shape)
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
    indices = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.float64)
    dense_shape = (3, 3)
    sp_a = tf.SparseTensor(indices, values, dense_shape)
    b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
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
    indices = np.array([[0, 1], [1, 0], [2, 2]], dtype=np.int64)
    values = np.array([4, 5, 6], dtype=np.int32)
    dense_shape = (3, 3)
    sp_a = tf.SparseTensor(indices, values, dense_shape)
    b = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=np.int32)
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
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    values = np.array([1, 2, 3, 4], dtype=np.float32)
    dense_shape = (2, 2)
    sp_a = tf.SparseTensor(indices, values, dense_shape)
    b = np.array([[5, 6], [7, 8]], dtype=np.float32)
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
    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([-1, 2], dtype=np.float32)
    dense_shape = (2, 3)
    sp_a = tf.SparseTensor(indices, values, dense_shape)
    b = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
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

    # Input 7 (different data type)
    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int32)
    dense_shape = (2, 3)
    sp_a = tf.SparseTensor(indices, values, dense_shape)
    b = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
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
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    values = np.array([1+1j, 2+2j], dtype=np.complex64)
    dense_shape = (2, 2)
    sp_a = tf.SparseTensor(indices, values, dense_shape)
    b = np.array([[3+3j, 4+4j], [5+5j, 6+6j]], dtype=np.complex64)
    adjoint_a = True
    adjoint_b = True
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
    indices = np.array([[0, 0], [1, 1], [2, 0]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.float32)
    dense_shape = (3, 2)
    sp_a = tf.SparseTensor(indices, values, dense_shape)
    b = np.array([[1,2,3],[4,5,6]], dtype=np.float32)
    adjoint_a = False
    adjoint_b = False
    name = "test9"

    input_dict = {
        "sp_a": sp_a,
        "b": b,
        "adjoint_a": adjoint_a,
        "adjoint_b": adjoint_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    
    # Input 11
    indices = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], dtype=np.int64)
    values = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    dense_shape = (2, 3)
    sp_a = tf.SparseTensor(indices, values, dense_shape)
    b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    adjoint_a = False
    adjoint_b = False
    name = "test11"

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
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.sparse_dense_matmul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.sparse_dense_matmul'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sparse.sparse_dense_matmul', generated_inputs['tf.sparse.sparse_dense_matmul'], lib="tf", suffix=0)
