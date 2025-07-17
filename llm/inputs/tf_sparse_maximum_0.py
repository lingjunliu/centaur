
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_maximum_inputs():
    list_of_inputs = []

    # Input 1
    indices_a = np.array([[0], [1]], dtype=np.int64)
    values_a = np.array([1, 2], dtype=np.int32)
    dense_shape_a = np.array([5], dtype=np.int64)
    sp_a = tf.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0], [2]], dtype=np.int64)
    values_b = np.array([3, 4], dtype=np.int32)
    dense_shape_b = np.array([5], dtype=np.int64)
    sp_b = tf.SparseTensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices_a = np.array([[0, 0], [0, 1]], dtype=np.int64)
    values_a = np.array([1, 2], dtype=np.int32)
    dense_shape_a = np.array([2, 2], dtype=np.int64)
    sp_a = tf.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 1], [1, 0]], dtype=np.int64)
    values_b = np.array([3, 4], dtype=np.int32)
    dense_shape_b = np.array([2, 2], dtype=np.int64)
    sp_b = tf.SparseTensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "max_sparse"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices_a = np.array([[0], [2], [4]], dtype=np.int64)
    values_a = np.array([-1, 0, 1], dtype=np.int32)
    dense_shape_a = np.array([5], dtype=np.int64)
    sp_a = tf.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[1], [2], [3]], dtype=np.int64)
    values_b = np.array([2, 0, -2], dtype=np.int32)
    dense_shape_b = np.array([5], dtype=np.int64)
    sp_b = tf.SparseTensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    indices_a = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int64)
    values_a = np.array([1, 2], dtype=np.int32)
    dense_shape_a = np.array([1, 2, 2], dtype=np.int64)
    sp_a = tf.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int64)
    values_b = np.array([3, 4], dtype=np.int32)
    dense_shape_b = np.array([1, 2, 2], dtype=np.int64)
    sp_b = tf.SparseTensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "sparse_max"}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    indices_a = np.array([[0], [1], [2]], dtype=np.int64)
    values_a = np.array([5, 2, 8], dtype=np.int32)
    dense_shape_a = np.array([10], dtype=np.int64)
    sp_a = tf.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[1], [3], [4]], dtype=np.int64)
    values_b = np.array([1, 9, 6], dtype=np.int32)
    dense_shape_b = np.array([10], dtype=np.int64)
    sp_b = tf.SparseTensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: different data type
    indices_a = np.array([[0], [1]], dtype=np.int64)
    values_a = np.array([1.0, 2.0], dtype=np.float32)
    dense_shape_a = np.array([5], dtype=np.int64)
    sp_a = tf.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0], [2]], dtype=np.int64)
    values_b = np.array([3.0, 4.0], dtype=np.float32)
    dense_shape_b = np.array([5], dtype=np.int64)
    sp_b = tf.SparseTensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "max_sparse_float"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D with float
    indices_a = np.array([[0, 0], [0, 1]], dtype=np.int64)
    values_a = np.array([1.0, 2.0], dtype=np.float32)
    dense_shape_a = np.array([2, 2], dtype=np.int64)
    sp_a = tf.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 1], [1, 0]], dtype=np.int64)
    values_b = np.array([3.0, 4.0], dtype=np.float32)
    dense_shape_b = np.array([2, 2], dtype=np.int64)
    sp_b = tf.SparseTensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger dense shape
    indices_a = np.array([[0], [1]], dtype=np.int64)
    values_a = np.array([1, 2], dtype=np.int32)
    dense_shape_a = np.array([10], dtype=np.int64)
    sp_a = tf.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0], [2]], dtype=np.int64)
    values_b = np.array([3, 4], dtype=np.int32)
    dense_shape_b = np.array([10], dtype=np.int64)
    sp_b = tf.SparseTensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "larger_dense"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative values and different dtypes
    indices_a = np.array([[0], [1]], dtype=np.int64)
    values_a = np.array([-1, 2], dtype=np.int64)
    dense_shape_a = np.array([5], dtype=np.int64)
    sp_a = tf.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0], [2]], dtype=np.int64)
    values_b = np.array([3, -4], dtype=np.int64)
    dense_shape_b = np.array([5], dtype=np.int64)
    sp_b = tf.SparseTensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sparse.maximum"] = tf_sparse_maximum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.maximum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.maximum'.")

check_valid('tf.sparse.maximum', generated_inputs['tf.sparse.maximum'], lib="tf", suffix=0)
