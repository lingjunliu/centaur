
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_add_inputs():
    list_of_inputs = []

    # Input 1: Sparse (int32) + Dense (int32)
    a1 = tf.SparseTensor(
        indices=np.array([[0, 1], [1, 2]], dtype=np.int64),
        values=np.array([1, 2], dtype=np.int32),
        dense_shape=np.array([3, 4], dtype=np.int64)
    )
    b1 = np.ones((3, 4), dtype=np.int32)
    threshold1 = np.array(0, dtype=np.int32)
    input_dict_1 = {'a': a1, 'b': b1, 'threshold': threshold1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Dense (float32) + Sparse (float32)
    a2 = np.random.rand(2, 2).astype(np.float32)
    b2 = tf.SparseTensor(
        indices=np.array([[0, 0], [1, 0]], dtype=np.int64),
        values=np.array([-0.9, 5.0], dtype=np.float32),
        dense_shape=np.array([2, 2], dtype=np.int64)
    )
    threshold2 = np.array(0.15, dtype=np.float32)
    input_dict_2 = {'a': a2, 'b': b2, 'threshold': threshold2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Sparse (int64) + Dense (int64)
    a3 = tf.SparseTensor(
        indices=np.array([[0, 1], [2, 2]], dtype=np.int64),
        values=np.array([10, 20], dtype=np.int64),
        dense_shape=np.array([3, 3], dtype=np.int64)
    )
    b3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
    threshold3 = np.array(0, dtype=np.int64)
    input_dict_3 = {'a': a3, 'b': b3, 'threshold': threshold3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Dense (float64) + Sparse (float64)
    a4 = np.ones((2, 5), dtype=np.float64)
    b4 = tf.SparseTensor(
        indices=np.array([[0, 0], [1, 3]], dtype=np.int64),
        values=np.array([-5.0, 8.0], dtype=np.float64),
        dense_shape=np.array([2, 5], dtype=np.int64)
    )
    threshold4 = np.array(0.0, dtype=np.float64)
    input_dict_4 = {'a': a4, 'b': b4, 'threshold': threshold4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Sparse tensor + dense tensor of zeros.
    a5 = tf.SparseTensor(
        indices=np.array([[1, 1], [2, 0]], dtype=np.int64),
        values=np.array([5, 8], dtype=np.int32),
        dense_shape=np.array([3, 2], dtype=np.int64)
    )
    b5 = np.zeros((3, 2), dtype=np.int32)
    threshold5 = np.array(0, dtype=np.int32)
    input_dict_5 = {'a': a5, 'b': b5, 'threshold': threshold5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Dense (int32) + Sparse (int32), with negative values
    a6 = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.int32)
    b6 = tf.SparseTensor(
        indices=np.array([[0, 0], [1, 1]], dtype=np.int64),
        values=np.array([1, 5], dtype=np.int32),
        dense_shape=np.array([2, 3], dtype=np.int64)
    )
    threshold6 = np.array(0, dtype=np.int32)
    input_dict_6 = {'a': a6, 'b': b6, 'threshold': threshold6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: Sparse (complex128) + Dense (complex128)
    a7 = tf.SparseTensor(
        indices=np.array([[0, 1]], dtype=np.int64),
        values=np.array([10+10j], dtype=np.complex128),
        dense_shape=np.array([2, 2], dtype=np.int64)
    )
    b7 = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
    threshold7 = np.array(0.0, dtype=np.float64)
    input_dict_7 = {'a': a7, 'b': b7, 'threshold': threshold7}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Empty SparseTensor + Dense Tensor
    a8 = tf.SparseTensor(
        indices=np.empty((0, 2), dtype=np.int64),
        values=np.array([], dtype=np.float32),
        dense_shape=np.array([3, 3], dtype=np.int64)
    )
    b8 = np.ones((3, 3), dtype=np.float32)
    threshold8 = np.array(0.0, dtype=np.float32)
    input_dict_8 = {'a': a8, 'b': b8, 'threshold': threshold8}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Sparse (float32) + Sparse (float32) -> Both are sparse, might fail analysis
    # This case is included to strictly test the API's functionality,
    # despite the potential incompatibility with the test harness.
    a9 = tf.SparseTensor(
        indices=np.array([[0, 0]], dtype=np.int64),
        values=np.array([1.0], dtype=np.float32),
        dense_shape=np.array([2, 2], dtype=np.int64)
    )
    b9 = tf.SparseTensor(
        indices=np.array([[1, 1]], dtype=np.int64),
        values=np.array([2.5], dtype=np.float32),
        dense_shape=np.array([2, 2], dtype=np.int64)
    )
    threshold9 = np.array(0.0, dtype=np.float32)
    input_dict_9 = {'a': a9, 'b': b9, 'threshold': threshold9}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Sparse (complex64) + Dense (complex64)
    a10 = tf.SparseTensor(
        indices=np.array([[0], [2]], dtype=np.int64),
        values=np.array([1+2j, 3-4j], dtype=np.complex64),
        dense_shape=np.array([4], dtype=np.int64)
    )
    b10 = np.zeros(4, dtype=np.complex64)
    threshold10 = np.array(0.0, dtype=np.float32)
    input_dict_10 = {'a': a10, 'b': b10, 'threshold': threshold10}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.sparse.add"] = tf_sparse_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.add' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.add'.")

check_valid('tf.sparse.add', generated_inputs['tf.sparse.add'], lib="tf", suffix=0)
