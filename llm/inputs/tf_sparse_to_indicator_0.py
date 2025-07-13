
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_to_indicator_inputs():
    list_of_inputs = []

    # Input 1
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values = np.array([0, 1, 2, 3], dtype=np.int64)
    dense_shape = np.array([2, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    vocab_size = 5
    name = "indicator_1"
    input_dict = {"sp_input": sp_input, "vocab_size": vocab_size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values = np.array([0, 1, 2, 3], dtype=np.int32)
    dense_shape = np.array([2, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    vocab_size = 4
    name = "indicator_2"
    input_dict = {"sp_input": sp_input, "vocab_size": vocab_size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = np.array([[0, 0], [0, 1], [0, 1]])
    values = np.array([0, 1, 2], dtype=np.int64)
    dense_shape = np.array([1, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    vocab_size = 3
    name = "indicator_3"
    input_dict = {"sp_input": sp_input, "vocab_size": vocab_size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: larger vocab_size
    indices = np.array([[0, 0], [0, 1]])
    values = np.array([1, 5], dtype=np.int32)
    dense_shape = np.array([1, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    vocab_size = 10
    name = "indicator_4"
    input_dict = {"sp_input": sp_input, "vocab_size": vocab_size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D input
    indices = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    values = np.array([2, 1, 0], dtype=np.int64)
    dense_shape = np.array([2, 2, 1])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    vocab_size = 3
    name = "indicator_5"
    input_dict = {"sp_input": sp_input, "vocab_size": vocab_size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6: Empty SparseTensor
    indices = np.array([], dtype=np.int64).reshape(0, 2)
    values = np.array([], dtype=np.int64)
    dense_shape = np.array([2, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    vocab_size = 5
    name = "indicator_6"
    input_dict = {"sp_input": sp_input, "vocab_size": vocab_size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different dense shape
    indices = np.array([[0, 0], [1, 1], [2, 2]])
    values = np.array([0, 1, 2], dtype=np.int32)
    dense_shape = np.array([3, 3])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    vocab_size = 4
    name = "indicator_7"
    input_dict = {"sp_input": sp_input, "vocab_size": vocab_size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: single dimension
    indices = np.array([[0], [1], [2]])
    values = np.array([0, 1, 2], dtype=np.int64)
    dense_shape = np.array([3])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    vocab_size = 3
    name = "indicator_8"
    input_dict = {"sp_input": sp_input, "vocab_size": vocab_size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: larger indices values
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values = np.array([1, 3, 2, 0], dtype=np.int32)
    dense_shape = np.array([2, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    vocab_size = 5
    name = "indicator_9"
    input_dict = {"sp_input": sp_input, "vocab_size": vocab_size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    def _sparse_size(sparse_tensor):
      return np.prod(sparse_tensor.dense_shape).numpy() if isinstance(sparse_tensor.dense_shape, tf.TensorShape) else np.prod(sparse_tensor.dense_shape)

    # Input 10: Reduced dimensions for indices to match dense shape
    indices = np.array([[0, 0], [0, 1], [1, 0]])
    values = np.array([0, 1, 2], dtype=np.int64)
    dense_shape = np.array([2, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    vocab_size = 3
    name = "indicator_10"
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
