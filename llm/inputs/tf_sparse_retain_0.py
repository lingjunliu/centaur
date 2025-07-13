
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_retain_inputs():
    list_of_inputs = []

    # Input 1
    indices = np.array([[0, 0], [1, 2]])
    values = np.array([1, 2])
    shape = np.array([3, 4])
    sp_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)
    to_retain = np.array([True, False])
    input_dict = {"sp_input": sp_input, "to_retain": to_retain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = np.array([[0, 1], [0, 3], [2, 0], [3, 1]])
    values = np.array(['a', 'b', 'c', 'd'])
    shape = np.array([4, 5])
    sp_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)
    to_retain = np.array([True, False, False, True])
    input_dict = {"sp_input": sp_input, "to_retain": to_retain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    values = np.array([1.0, 2.0, 3.0])
    shape = np.array([3, 3, 3])
    sp_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)
    to_retain = np.array([False, True, True])
    input_dict = {"sp_input": sp_input, "to_retain": to_retain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    indices = np.array([[0, 1], [1, 0], [2, 2]])
    values = np.array([1, -2, 3])
    shape = np.array([3, 3])
    sp_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)
    to_retain = np.array([True, True, False])
    input_dict = {"sp_input": sp_input, "to_retain": to_retain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values = np.array([1, 2, 3, 4])
    shape = np.array([2, 2])
    sp_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)
    to_retain = np.array([True, True, True, True])
    input_dict = {"sp_input": sp_input, "to_retain": to_retain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values = np.array([1, 2, 3, 4])
    shape = np.array([2, 2])
    sp_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)
    to_retain = np.array([False, False, False, False])
    input_dict = {"sp_input": sp_input, "to_retain": to_retain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    indices = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]])
    values = np.array([1, 2, 3, 4])
    shape = np.array([1, 2, 2])
    sp_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)
    to_retain = np.array([True, False, True, False])
    input_dict = {"sp_input": sp_input, "to_retain": to_retain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    indices = np.array([[0, 0], [1, 1], [2, 2]])
    values = np.array([5, 6, 7])
    shape = np.array([3, 3])
    sp_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)
    to_retain = np.array([True, True, True])
    input_dict = {"sp_input": sp_input, "to_retain": to_retain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    indices = np.array([[0, 0], [1, 1], [2, 2]])
    values = np.array([5, 6, 7])
    shape = np.array([3, 3])
    sp_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)
    to_retain = np.array([False, False, False])
    input_dict = {"sp_input": sp_input, "to_retain": to_retain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    indices = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    values = np.array([10, 20, 30, 40])
    shape = np.array([9, 9])
    sp_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)
    to_retain = np.array([True, False, True, False])
    input_dict = {"sp_input": sp_input, "to_retain": to_retain}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.retain"] = tf_sparse_retain_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.retain' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.retain'.")

check_valid('tf.sparse.retain', generated_inputs['tf.sparse.retain'], lib="tf", suffix=0)
