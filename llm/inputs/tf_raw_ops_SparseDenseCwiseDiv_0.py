
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseDenseCwiseDiv_inputs():
    list_of_inputs = []

    # Input 1
    sp_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    sp_values = np.array([1.0, 2.0], dtype=np.float32)
    sp_shape = np.array([2, 3], dtype=np.int64)
    dense = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    name = "sparse_dense_cwise_div_1"
    input_dict = {"sp_indices": tf.constant(sp_indices), "sp_values": tf.constant(sp_values), "sp_shape": tf.constant(sp_shape), "dense": tf.constant(dense), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    sp_indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
    sp_values = np.array([5, 6], dtype=np.int32)
    sp_shape = np.array([2, 2], dtype=np.int64)
    dense = np.array([[1, 2], [3, 4]], dtype=np.int32)
    name = "sparse_dense_cwise_div_2"
    input_dict = {"sp_indices": tf.constant(sp_indices), "sp_values": tf.constant(sp_values), "sp_shape": tf.constant(sp_shape), "dense": tf.constant(dense), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    sp_indices = np.array([[0, 0, 0], [0, 1, 1]], dtype=np.int64)
    sp_values = np.array([7.0, 8.0], dtype=np.float64)
    sp_shape = np.array([1, 2, 2], dtype=np.int64)
    dense = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float64)
    name = "sparse_dense_cwise_div_3"
    input_dict = {"sp_indices": tf.constant(sp_indices), "sp_values": tf.constant(sp_values), "sp_shape": tf.constant(sp_shape), "dense": tf.constant(dense), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    sp_indices = np.array([[0], [1]], dtype=np.int64)
    sp_values = np.array([9, 10], dtype=np.int32)
    sp_shape = np.array([2], dtype=np.int64)
    dense = np.array([1, 2], dtype=np.int32)
    name = "sparse_dense_cwise_div_4"
    input_dict = {"sp_indices": tf.constant(sp_indices), "sp_values": tf.constant(sp_values), "sp_shape": tf.constant(sp_shape), "dense": tf.constant(dense), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    sp_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    sp_values = np.array([11, 12], dtype=np.int32)
    sp_shape = np.array([2, 2], dtype=np.int64)
    dense = np.array([[2, 3], [4, 5]], dtype=np.int32)
    name = "sparse_dense_cwise_div_5"
    input_dict = {"sp_indices": tf.constant(sp_indices), "sp_values": tf.constant(sp_values), "sp_shape": tf.constant(sp_shape), "dense": tf.constant(dense), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    sp_indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    sp_values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    sp_shape = np.array([2, 2], dtype=np.int64)
    dense = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    name = "sparse_dense_cwise_div_6"
    input_dict = {"sp_indices": tf.constant(sp_indices), "sp_values": tf.constant(sp_values), "sp_shape": tf.constant(sp_shape), "dense": tf.constant(dense), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    sp_indices = np.array([[0, 0, 0]], dtype=np.int64)
    sp_values = np.array([1.0], dtype=np.float32)
    sp_shape = np.array([1, 1, 1], dtype=np.int64)
    dense = np.array([[[2.0]]], dtype=np.float32)
    name = "sparse_dense_cwise_div_7"
    input_dict = {"sp_indices": tf.constant(sp_indices), "sp_values": tf.constant(sp_values), "sp_shape": tf.constant(sp_shape), "dense": tf.constant(dense), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    sp_indices = np.array([[0, 0], [1, 0]], dtype=np.int64)
    sp_values = np.array([-1.0, -2.0], dtype=np.float32)
    sp_shape = np.array([2, 1], dtype=np.int64)
    dense = np.array([[-3.0], [-4.0]], dtype=np.float32)
    name = "sparse_dense_cwise_div_8"
    input_dict = {"sp_indices": tf.constant(sp_indices), "sp_values": tf.constant(sp_values), "sp_shape": tf.constant(sp_shape), "dense": tf.constant(dense), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    sp_indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    sp_values = np.array([1, -2], dtype=np.int32)
    sp_shape = np.array([1, 2], dtype=np.int64)
    dense = np.array([[3, -4]], dtype=np.int32)
    name = "sparse_dense_cwise_div_9"
    input_dict = {"sp_indices": tf.constant(sp_indices), "sp_values": tf.constant(sp_values), "sp_shape": tf.constant(sp_shape), "dense": tf.constant(dense), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    sp_indices = np.array([[0, 0, 0, 0]], dtype=np.int64)
    sp_values = np.array([2.0], dtype=np.float32)
    sp_shape = np.array([1, 1, 1, 1], dtype=np.int64)
    dense = np.array([[[[3.0]]]], dtype=np.float32)
    name = "sparse_dense_cwise_div_10"
    input_dict = {"sp_indices": tf.constant(sp_indices), "sp_values": tf.constant(sp_values), "sp_shape": tf.constant(sp_shape), "dense": tf.constant(dense), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 - Empty SparseTensor
    sp_indices = np.array([], dtype=np.int64).reshape(0, 2)
    sp_values = np.array([], dtype=np.float32)
    sp_shape = np.array([2, 2], dtype=np.int64)
    dense = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    name = "sparse_dense_cwise_div_11"
    input_dict = {"sp_indices": tf.constant(sp_indices), "sp_values": tf.constant(sp_values), "sp_shape": tf.constant(sp_shape), "dense": tf.constant(dense), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 - broadcast dense to sparse
    sp_indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    sp_values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    sp_shape = np.array([2, 2], dtype=np.int64)
    dense = np.array([2.0], dtype=np.float32)
    name = "sparse_dense_cwise_div_12"
    input_dict = {"sp_indices": tf.constant(sp_indices), "sp_values": tf.constant(sp_values), "sp_shape": tf.constant(sp_shape), "dense": tf.constant(dense), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13 - different dtypes
    sp_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    sp_values = np.array([1, 2], dtype=np.int64)
    sp_shape = np.array([2, 3], dtype=np.int64)
    dense = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    name = "sparse_dense_cwise_div_13"
    input_dict = {"sp_indices": tf.constant(sp_indices), "sp_values": tf.constant(sp_values), "sp_shape": tf.constant(sp_shape), "dense": tf.constant(dense), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseDenseCwiseDiv"] = tf_raw_ops_SparseDenseCwiseDiv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseDenseCwiseDiv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseDenseCwiseDiv'.")

check_valid('tf.raw_ops.SparseDenseCwiseDiv', generated_inputs['tf.raw_ops.SparseDenseCwiseDiv'], lib="tf", suffix=0)
