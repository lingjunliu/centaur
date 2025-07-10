
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_expand_dims_inputs():
    list_of_inputs = []

    def to_numpy(tensor):
        return tensor.to_dense().numpy()

    # Input 1
    indices = np.array([[0, 0], [1, 2]])
    values = np.array([1, 2])
    dense_shape = np.array([3, 4])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    axis = 0
    name = "sparse_expanded_1"
    input_dict = {"sp_input": sp_input, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = np.array([[0, 0], [1, 1], [2, 2]])
    values = np.array([1.0, 2.0, 3.0])
    dense_shape = np.array([3, 3])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    axis = 1
    name = "sparse_expanded_2"
    input_dict = {"sp_input": sp_input, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = np.array([[0, 0, 0], [1, 1, 1]])
    values = np.array([1, 2])
    dense_shape = np.array([2, 2, 2])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    axis = -1
    name = "sparse_expanded_3"
    input_dict = {"sp_input": sp_input, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    indices = np.array([[0, 0], [0, 1]])
    values = np.array([1, 2])
    dense_shape = np.array([1, 2])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    axis = -2
    name = "sparse_expanded_4"
    input_dict = {"sp_input": sp_input, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    indices = np.array([[0, 0, 0, 0]])
    values = np.array([1])
    dense_shape = np.array([1, 1, 1, 1])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    axis = 0
    name = "sparse_expanded_5"
    input_dict = {"sp_input": sp_input, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    indices = np.array([[1, 1]])
    values = np.array([5])
    dense_shape = np.array([5, 5])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    axis = 0
    name = "sparse_expanded_6"
    input_dict = {"sp_input": sp_input, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    indices = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
    values = np.array([1, 2, 3, 4])
    dense_shape = np.array([2, 2, 1])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    axis = 1
    name = "sparse_expanded_7"
    input_dict = {"sp_input": sp_input, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    indices = np.array([[0], [1]])
    values = np.array([1, 2])
    dense_shape = np.array([2])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    axis = 0
    name = "sparse_expanded_8"
    input_dict = {"sp_input": sp_input, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.expand_dims"] = tf_sparse_expand_dims_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.expand_dims' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.expand_dims'.")

check_valid('tf.sparse.expand_dims', generated_inputs['tf.sparse.expand_dims'], lib="tf", suffix=0)
