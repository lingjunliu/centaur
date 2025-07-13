
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_reorder_inputs():
    list_of_inputs = []

    def create_sparse_tensor(indices, values, shape):
        return tf.SparseTensor(indices=indices, values=values, dense_shape=shape)

    def get_tensor_size(tensor):
        return np.prod(tensor.dense_shape)  # Use dense_shape for size

    # Input 1
    indices = np.array([[0, 1], [0, 0], [1, 2], [1, 1]], dtype=np.int64)
    values = np.array([1, 2, 3, 4], dtype=np.float32)
    shape = np.array([2, 3], dtype=np.int64)
    sp_input = create_sparse_tensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_sparse_tensor_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = np.array([[0, 2], [0, 0], [1, 1]], dtype=np.int64)
    values = np.array([5, 6, 7], dtype=np.int32)
    shape = np.array([2, 3], dtype=np.int64)
    sp_input = create_sparse_tensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_sparse_tensor_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = np.array([[2, 1], [0, 0], [1, 0], [0, 1]], dtype=np.int64)
    values = np.array([8, 9, 10, 11], dtype=np.float64)
    shape = np.array([3, 3], dtype=np.int64)
    sp_input = create_sparse_tensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_sparse_tensor_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4
    indices = np.array([[0, 0, 1], [0, 0, 0], [0, 1, 0]], dtype=np.int64)
    values = np.array([12, 13, 14], dtype=np.int32)
    shape = np.array([1, 2, 2], dtype=np.int64)
    sp_input = create_sparse_tensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_sparse_tensor_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    indices = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 1]], dtype=np.int64)
    values = np.array([15, 16, 17], dtype=np.float32)
    shape = np.array([2, 2, 2], dtype=np.int64)
    sp_input = create_sparse_tensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_sparse_tensor_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    indices = np.array([[0, 1], [1, 0], [0, 0]], dtype=np.int64)
    values = np.array([18, 19, 20], dtype=np.int64)
    shape = np.array([2, 2], dtype=np.int64)
    sp_input = create_sparse_tensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_sparse_tensor_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    values = np.array([21, 22, 23], dtype=np.float32)
    shape = np.array([2, 2], dtype=np.int64)
    sp_input = create_sparse_tensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_sparse_tensor_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.reorder"] = tf_sparse_reorder_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.reorder' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.reorder'.")

check_valid('tf.sparse.reorder', generated_inputs['tf.sparse.reorder'], lib="tf", suffix=0)
