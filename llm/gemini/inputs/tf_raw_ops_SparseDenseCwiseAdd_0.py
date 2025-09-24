
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseDenseCwiseAdd_inputs():
    list_of_inputs = []

    # Input 1
    sp_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    sp_values = np.array([1.0, 2.0], dtype=np.float32)
    sp_shape = np.array([2, 3], dtype=np.int64)
    dense = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": "add1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    sp_indices = np.array([[0, 1], [1, 0], [1, 1]], dtype=np.int64)
    sp_values = np.array([1, 2, 3], dtype=np.int32)
    sp_shape = np.array([2, 2], dtype=np.int64)
    dense = np.array([[4, 5], [6, 7]], dtype=np.int32)
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": "add2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    sp_indices = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int64)
    sp_values = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    sp_shape = np.array([2, 2, 2], dtype=np.int64)
    dense = np.ones((2, 2, 2), dtype=np.float64) * 0.5
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": "add3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    sp_indices = np.array([[0], [1], [2]], dtype=np.int64)
    sp_values = np.array([1, 2, 3], dtype=np.int64)
    sp_shape = np.array([4], dtype=np.int64)
    dense = np.array([4, 5, 6, 7], dtype=np.int64)
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": "add4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    sp_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    sp_values = np.array([1, 2], dtype=np.int8)
    sp_shape = np.array([2, 2], dtype=np.int64)
    dense = np.array([[3, 4], [5, 6]], dtype=np.int8)
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": "add5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    sp_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    sp_values = np.array([1.0, 2.0], dtype=np.float32)
    sp_shape = np.array([2, 2], dtype=np.int64)
    dense = np.array([1.0], dtype=np.float32)
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": "add7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (empty sparse tensor)
    sp_indices = np.empty((0, 2), dtype=np.int64)
    sp_values = np.array([], dtype=np.float32)
    sp_shape = np.array([2, 2], dtype=np.int64)
    dense = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": "add8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, different shape for broadcasting
    sp_indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    sp_values = np.array([1.0, 2.0], dtype=np.float32)
    sp_shape = np.array([1, 2], dtype=np.int64)
    dense = np.array([1.0, 2.0], dtype=np.float32)
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": "add9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, larger dimensions
    sp_indices = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int64)
    sp_values = np.array([1.0, 2.0], dtype=np.float32)
    sp_shape = np.array([2, 2, 2, 2], dtype=np.int64)
    dense = np.ones((2, 2, 2, 2), dtype=np.float32) * 0.5
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": "add10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseDenseCwiseAdd"] = tf_raw_ops_SparseDenseCwiseAdd_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseDenseCwiseAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseDenseCwiseAdd'.")

check_valid('tf.raw_ops.SparseDenseCwiseAdd', generated_inputs['tf.raw_ops.SparseDenseCwiseAdd'], lib="tf", suffix=0)
