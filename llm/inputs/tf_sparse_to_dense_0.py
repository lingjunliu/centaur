
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_to_dense_inputs():
    list_of_inputs = []

    # Input 1
    indices = np.array([[0, 1], [1, 0], [2, 2]])
    values = np.array([1, 2, 3], dtype=np.int32)
    dense_shape = np.array([3, 3])
    sp_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    default_value = tf.constant(0, dtype=np.int32)
    validate_indices = True
    name = "sparse_to_dense_1"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "validate_indices": validate_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = np.array([[0, 0], [1, 1], [2, 2], [0, 2]])
    values = np.array([4, 5, 6, 7], dtype=np.float32)
    dense_shape = np.array([3, 3])
    sp_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    default_value = tf.constant(-1.0, dtype=np.float32)
    validate_indices = False
    name = "sparse_to_dense_2"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "validate_indices": validate_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1]])
    values = np.array([10, 11, 12], dtype=np.int64)
    dense_shape = np.array([2, 2, 2])
    sp_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    default_value = tf.constant(0, dtype=np.int64)
    validate_indices = True
    name = "sparse_to_dense_3"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "validate_indices": validate_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    dense_shape = np.array([2, 2])
    sp_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    default_value = tf.constant(0.0, dtype=np.float64)
    validate_indices = False
    name = "sparse_to_dense_4"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "validate_indices": validate_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    indices = np.array([[0, 0, 0], [1, 1, 1]])
    values = np.array([-1, -2], dtype=np.int32)
    dense_shape = np.array([2, 2, 2])
    sp_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    default_value = tf.constant(100, dtype=np.int32)
    validate_indices = True
    name = "sparse_to_dense_5"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "validate_indices": validate_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.to_dense"] = tf_sparse_to_dense_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.to_dense' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.to_dense'.")

check_valid('tf.sparse.to_dense', generated_inputs['tf.sparse.to_dense'], lib="tf", suffix=0)
