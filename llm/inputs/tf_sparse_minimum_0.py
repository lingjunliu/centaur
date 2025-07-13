
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_minimum_inputs():
    list_of_inputs = []

    def create_sparse_tensor(indices, values, dense_shape):
        return tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

    def is_valid_sparse_tensor(sparse_tensor):
        return isinstance(sparse_tensor, tf.sparse.SparseTensor)

    # Input 1
    sp_a = create_sparse_tensor(np.array([[0, 0], [1, 2]]), np.array([1, 2], dtype=np.int32), np.array([3, 4]))
    sp_b = create_sparse_tensor(np.array([[0, 0], [1, 2]]), np.array([3, 4], dtype=np.int32), np.array([3, 4]))
    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "minimum_1"}
    if is_valid_sparse_tensor(sp_a) and is_valid_sparse_tensor(sp_b):
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    sp_a = create_sparse_tensor(np.array([[0], [2]]), np.array([5, 6], dtype=np.int32), np.array([5]))
    sp_b = create_sparse_tensor(np.array([[0], [2]]), np.array([7, 8], dtype=np.int32), np.array([5]))
    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "minimum_2"}
    if is_valid_sparse_tensor(sp_a) and is_valid_sparse_tensor(sp_b):
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3 (negative values)
    sp_a = create_sparse_tensor(np.array([[0, 1], [2, 3]]), np.array([-1, -2], dtype=np.int32), np.array([4, 5]))
    sp_b = create_sparse_tensor(np.array([[0, 1], [2, 3]]), np.array([0, -3], dtype=np.int32), np.array([4, 5]))
    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "minimum_3"}
    if is_valid_sparse_tensor(sp_a) and is_valid_sparse_tensor(sp_b):
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 (different values)
    sp_a = create_sparse_tensor(np.array([[0, 0], [1, 1], [2, 2]]), np.array([10, 20, 30], dtype=np.int32), np.array([3, 3]))
    sp_b = create_sparse_tensor(np.array([[0, 0], [1, 1], [2, 2]]), np.array([5, 25, 35], dtype=np.int32), np.array([3, 3]))
    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "minimum_4"}
    if is_valid_sparse_tensor(sp_a) and is_valid_sparse_tensor(sp_b):
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (single element sparse tensor)
    sp_a = create_sparse_tensor(np.array([[0]]), np.array([1], dtype=np.int32), np.array([1]))
    sp_b = create_sparse_tensor(np.array([[0]]), np.array([2], dtype=np.int32), np.array([1]))
    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "minimum_5"}
    if is_valid_sparse_tensor(sp_a) and is_valid_sparse_tensor(sp_b):
        list_of_inputs.append(copy.deepcopy(input_dict))
        
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.minimum"] = tf_sparse_minimum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.minimum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.minimum'.")

check_valid('tf.sparse.minimum', generated_inputs['tf.sparse.minimum'], lib="tf", suffix=0)
