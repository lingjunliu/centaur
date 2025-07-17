
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_fill_empty_rows_inputs():
    list_of_inputs = []

    # Input 1: Basic case with int32
    indices = np.array([[0, 1], [0, 3], [2, 0], [3, 1]], dtype=np.int64)
    values = np.array([1, 2, 3, 4], dtype=np.int32)
    shape = np.array([5, 6], dtype=np.int64)
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    default_value = tf.constant(5, dtype=tf.int32)
    name = "test_sparse_fill_1"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Empty rows at the beginning
    indices = np.array([[2, 0], [3, 1]], dtype=np.int64)
    values = np.array([1.0, 2.0], dtype=np.float32)
    shape = np.array([5, 6], dtype=np.int64)
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    default_value = tf.constant(0.0, dtype=tf.float32)
    name = "test_sparse_fill_2"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty rows at the end
    indices = np.array([[0, 1], [0, 3], [2, 0]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    shape = np.array([5, 6], dtype=np.int64)
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    default_value = tf.constant(0, dtype=tf.int32)
    name = "test_sparse_fill_3"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: All rows are empty
    indices = np.array([], dtype=np.int64).reshape(0, 2)
    values = np.array([], dtype=np.int32)
    shape = np.array([5, 6], dtype=np.int64)
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    default_value = tf.constant(10, dtype=tf.int32)
    name = "test_sparse_fill_4"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: No empty rows
    indices = np.array([[0, 1], [1, 3], [2, 0], [3, 1], [4, 2]], dtype=np.int64)
    values = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    shape = np.array([5, 6], dtype=np.int64)
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    default_value = tf.constant(-1, dtype=tf.int32)
    name = "test_sparse_fill_5"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different default value type (float64)
    indices = np.array([[0, 1], [0, 3], [2, 0], [3, 1]], dtype=np.int64)
    values = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    shape = np.array([5, 6], dtype=np.int64)
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    default_value = tf.constant(0.0, dtype=tf.float64)
    name = "test_sparse_fill_6"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different shape
    indices = np.array([[0, 1], [2, 0]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int32)
    shape = np.array([3, 2], dtype=np.int64)
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    default_value = tf.constant(0, dtype=tf.int32)
    name = "test_sparse_fill_7"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Bool type
    indices = np.array([[0, 1], [0, 3], [2, 0]], dtype=np.int64)
    values = np.array([True, False, True], dtype=np.bool_)
    shape = np.array([5, 6], dtype=np.int64)
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    default_value = tf.constant(False, dtype=tf.bool)
    name = "test_sparse_fill_8"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large row and column indices
    indices = np.array([[0, 1000], [2, 0]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int32)
    shape = np.array([5, 2000], dtype=np.int64)
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    default_value = tf.constant(0, dtype=tf.int32)
    name = "test_sparse_fill_9"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Int64 type
    indices = np.array([[0, 1], [0, 3], [2, 0], [3, 1]], dtype=np.int64)
    values = np.array([1, 2, 3, 4], dtype=np.int64)
    shape = np.array([5, 6], dtype=np.int64)
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    default_value = tf.constant(0, dtype=tf.int64)
    name = "test_sparse_fill_10"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Rank 2 SparseTensor with int8 values
    indices = np.array([[0, 1], [2, 0]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int8)
    shape = np.array([5, 6], dtype=np.int64)
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    default_value = tf.constant(0, dtype=tf.int8)
    name = "test_sparse_fill_11"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sparse.fill_empty_rows"] = tf_sparse_fill_empty_rows_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.fill_empty_rows' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.fill_empty_rows'.")

check_valid('tf.sparse.fill_empty_rows', generated_inputs['tf.sparse.fill_empty_rows'], lib="tf", suffix=0)
