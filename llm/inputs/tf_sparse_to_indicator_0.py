
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_to_indicator_inputs():
    list_of_inputs = []

    def get_sparse_tensor_size(sp_input):
        return np.prod(tf.sparse.to_dense(sp_input).shape.numpy())

    # Input 1: Basic valid input
    indices = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 3], [1, 1, 1]])
    values = np.array([0, 10, 103, 150], dtype=np.int64)
    dense_shape = np.array([2, 2, 4], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    vocab_size = 200
    name = "indicator_tensor_1"
    input_dict = {"sp_input": sp_input, "vocab_size": vocab_size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shape, smaller vocab_size
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values = np.array([0, 1, 2, 3], dtype=np.int32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    vocab_size = 5
    name = "indicator_tensor_2"
    input_dict = {"sp_input": sp_input, "vocab_size": vocab_size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: One dimensional SparseTensor
    indices = np.array([[0], [1], [2], [3]])
    values = np.array([1, 5, 2, 8], dtype=np.int64)
    dense_shape = np.array([4], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    vocab_size = 10
    name = "indicator_tensor_3"
    input_dict = {"sp_input": sp_input, "vocab_size": vocab_size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Three dimensional SparseTensor
    indices = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 2],
        [1, 1, 0],
        [1, 1, 1],
    ])
    values = np.array([5, 3, 7, 9, 2], dtype=np.int32)
    dense_shape = np.array([2, 2, 3], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    vocab_size = 10
    name = "indicator_tensor_4"
    input_dict = {"sp_input": sp_input, "vocab_size": vocab_size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5:  vocab_size = 1
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values = np.array([0, 0, 0, 0], dtype=np.int32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    vocab_size = 1
    name = "indicator_tensor_5"
    input_dict = {"sp_input": sp_input, "vocab_size": vocab_size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6:  Larger vocab size
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values = np.array([100, 200, 300, 400], dtype=np.int32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    vocab_size = 500
    name = "indicator_tensor_6"
    input_dict = {"sp_input": sp_input, "vocab_size": vocab_size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7:  Empty SparseTensor
    indices = np.array([], dtype=np.int64).reshape(0, 2)
    values = np.array([], dtype=np.int32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    vocab_size = 10
    name = "indicator_tensor_7"
    input_dict = {"sp_input": sp_input, "vocab_size": vocab_size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8:  One element SparseTensor
    indices = np.array([[0, 0]], dtype=np.int64)
    values = np.array([5], dtype=np.int32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    vocab_size = 10
    name = "indicator_tensor_8"
    input_dict = {"sp_input": sp_input, "vocab_size": vocab_size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: large number of elements
    indices = np.array([[i, 0] for i in range(100)], dtype=np.int64)
    values = np.array([i for i in range(100)], dtype=np.int32)
    dense_shape = np.array([100, 1], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    vocab_size = 101
    name = "indicator_tensor_9"
    input_dict = {"sp_input": sp_input, "vocab_size": vocab_size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 2D, large vocab size, int64 values
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    values = np.array([100000, 200000, 300000, 400000], dtype=np.int64)
    dense_shape = np.array([2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    vocab_size = 500000
    name = "indicator_tensor_10"
    input_dict = {"sp_input": sp_input, "vocab_size": vocab_size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: with name None
    indices = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 3], [1, 1, 1]])
    values = np.array([0, 10, 103, 150], dtype=np.int64)
    dense_shape = np.array([2, 2, 4], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    vocab_size = 200
    name = None
    input_dict = {"sp_input": sp_input, "vocab_size": vocab_size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    #Input 12: sp_input value is tf.constant
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values = tf.constant([0, 1, 2, 3], dtype=tf.int32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    vocab_size = 5
    name = "indicator_tensor_12"
    input_dict = {"sp_input": sp_input, "vocab_size": vocab_size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    #Input 13: dense_shape is tf.constant
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values = np.array([0, 1, 2, 3], dtype=np.int32)
    dense_shape = tf.constant([2, 2], dtype=tf.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    vocab_size = 5
    name = "indicator_tensor_13"
    input_dict = {"sp_input": sp_input, "vocab_size": vocab_size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.to_indicator"] = tf_sparse_to_indicator_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.to_indicator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.to_indicator'.")

check_valid('tf.sparse.to_indicator', generated_inputs['tf.sparse.to_indicator'], lib="tf", suffix=0)
