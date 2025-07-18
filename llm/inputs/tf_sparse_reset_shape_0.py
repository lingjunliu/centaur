
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_reset_shape_inputs():
    list_of_inputs = []

    # Input 1
    sp_input = tf.SparseTensor(
        indices=np.array([[0, 1], [1, 2]], dtype=np.int64),
        values=np.array([1, 2], dtype=np.int32),
        dense_shape=np.array([2, 4], dtype=np.int64)
    )
    new_shape = np.array([3, 5], dtype=np.int64)
    input_dict = {'sp_input': sp_input, 'new_shape': new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    sp_input = tf.SparseTensor(
        indices=np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64),
        values=np.array([1.5, 2.5, 3.5], dtype=np.float32),
        dense_shape=np.array([4, 4], dtype=np.int64)
    )
    new_shape = np.array([4, 4], dtype=np.int64)
    input_dict = {'sp_input': sp_input, 'new_shape': new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    sp_input = tf.SparseTensor(
        indices=np.array([[0, 1, 2], [1, 0, 1]], dtype=np.int64),
        values=np.array([11, 22], dtype=np.int32),
        dense_shape=np.array([2, 2, 3], dtype=np.int64)
    )
    new_shape = np.array([3, 4, 5], dtype=np.int64)
    input_dict = {'sp_input': sp_input, 'new_shape': new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty sparse tensor
    sp_input = tf.SparseTensor(
        indices=np.empty((0, 2), dtype=np.int64),
        values=np.array([], dtype=np.int32),
        dense_shape=np.array([5, 5], dtype=np.int64)
    )
    new_shape = np.array([6, 7], dtype=np.int64)
    input_dict = {'sp_input': sp_input, 'new_shape': new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int64 values
    sp_input = tf.SparseTensor(
        indices=np.array([[0, 0], [9, 9]], dtype=np.int64),
        values=np.array([100, 200], dtype=np.int64),
        dense_shape=np.array([10, 10], dtype=np.int64)
    )
    new_shape = np.array([12, 12], dtype=np.int64)
    input_dict = {'sp_input': sp_input, 'new_shape': new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4-D tensor
    sp_input = tf.SparseTensor(
        indices=np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int64),
        values=np.array([1.0, 2.0], dtype=np.float32),
        dense_shape=np.array([2, 2, 2, 2], dtype=np.int64)
    )
    new_shape = np.array([3, 3, 3, 3], dtype=np.int64)
    input_dict = {'sp_input': sp_input, 'new_shape': new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty 3-D tensor
    sp_input = tf.SparseTensor(
        indices=np.empty((0, 3), dtype=np.int64),
        values=np.array([], dtype=np.float32),
        dense_shape=np.array([1, 2, 3], dtype=np.int64)
    )
    new_shape = np.array([2, 3, 4], dtype=np.int64)
    input_dict = {'sp_input': sp_input, 'new_shape': new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single element sparse tensor
    sp_input = tf.SparseTensor(
        indices=np.array([[0, 0]], dtype=np.int64),
        values=np.array([42], dtype=np.int32),
        dense_shape=np.array([1, 1], dtype=np.int64)
    )
    new_shape = np.array([100, 200], dtype=np.int64)
    input_dict = {'sp_input': sp_input, 'new_shape': new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1-D tensor
    sp_input = tf.SparseTensor(
        indices=np.array([[1], [3], [5]], dtype=np.int64),
        values=np.array([1, 3, 5], dtype=np.int32),
        dense_shape=np.array([10], dtype=np.int64)
    )
    new_shape = np.array([15], dtype=np.int64)
    input_dict = {'sp_input': sp_input, 'new_shape': new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64 values
    sp_input = tf.SparseTensor(
        indices=np.array([[0, 0, 0]], dtype=np.int64),
        values=np.array([3.14], dtype=np.float64),
        dense_shape=np.array([2, 2, 2], dtype=np.int64)
    )
    new_shape = np.array([3, 3, 3], dtype=np.int64)
    input_dict = {'sp_input': sp_input, 'new_shape': new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.sparse.reset_shape"] = tf_sparse_reset_shape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.reset_shape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.reset_shape'.")

check_valid('tf.sparse.reset_shape', generated_inputs['tf.sparse.reset_shape'], lib="tf", suffix=0)
