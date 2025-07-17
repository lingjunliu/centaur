
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_reorder_inputs():
    list_of_inputs = []

    # Input 1: Basic reordering
    indices = np.array([[0, 3], [0, 1], [3, 1], [2, 0]])
    values = np.array([1, 2, 3, 4])
    shape = np.array([4, 5])
    sp_input = tf.SparseTensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_tensor_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Already ordered
    indices = np.array([[0, 1], [0, 3], [2, 0], [3, 1]])
    values = np.array([5, 6, 7, 8])
    shape = np.array([4, 5])
    sp_input = tf.SparseTensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_tensor_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different shape
    indices = np.array([[1, 0], [0, 1], [2, 2]])
    values = np.array([1, 2, 3])
    shape = np.array([3, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_tensor_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Higher dimensions
    indices = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    values = np.array([4, 5, 6])
    shape = np.array([2, 2, 2])
    sp_input = tf.SparseTensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_tensor_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Integer values
    indices = np.array([[0, 2], [1, 0], [1, 1]])
    values = np.array([7, 8, 9], dtype=np.int64)
    shape = np.array([2, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_tensor_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float values
    indices = np.array([[0, 2], [1, 0], [1, 1]])
    values = np.array([7.1, 8.2, 9.3], dtype=np.float32)
    shape = np.array([2, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_tensor_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty SparseTensor (all values are zero by default, so technically it's already ordered)
    indices = np.empty((0, 2), dtype=np.int64)
    values = np.array([], dtype=np.int64)
    shape = np.array([2, 2])
    sp_input = tf.SparseTensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_tensor_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: One element only
    indices = np.array([[0, 0]])
    values = np.array([10])
    shape = np.array([1, 1])
    sp_input = tf.SparseTensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_tensor_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger Shape, More elements
    indices = np.array([[0, 1], [0, 5], [2, 2], [3, 0], [4, 4]])
    values = np.array([11, 12, 13, 14, 15])
    shape = np.array([5, 6])
    sp_input = tf.SparseTensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_tensor_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D, different ordering
    indices = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    values = np.array([16, 17, 18])
    shape = np.array([2, 2, 2])
    sp_input = tf.SparseTensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_tensor_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Unsorted indices in last dimension
    indices = np.array([[0, 0, 2], [0, 0, 0], [0, 0, 1]])
    values = np.array([1, 2, 3])
    shape = np.array([1, 1, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_tensor_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Larger 3D Shape
    indices = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [2, 0, 2]])
    values = np.array([1, 2, 3, 4, 5])
    shape = np.array([3, 2, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_tensor_12"}
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
