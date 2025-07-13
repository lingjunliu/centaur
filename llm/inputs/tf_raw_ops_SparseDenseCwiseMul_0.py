
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sparse_dense_cwise_mul_inputs():
    list_of_inputs = []

    # Input 1: Basic example with float32
    sp_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    sp_values = np.array([1.0, 2.0], dtype=np.float32)
    sp_shape = np.array([2, 3], dtype=np.int64)
    dense = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    name = "mul1"
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Example with int32
    sp_indices = np.array([[0, 1], [2, 0]], dtype=np.int64)
    sp_values = np.array([5, 6], dtype=np.int32)
    sp_shape = np.array([3, 2], dtype=np.int64)
    dense = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    name = "mul2"
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Example with float64 and different shape
    sp_indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    sp_values = np.array([2.5, 3.5], dtype=np.float64)
    sp_shape = np.array([2, 2, 2], dtype=np.int64)
    dense = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    name = "mul3"
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Example with uint8
    sp_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    sp_values = np.array([7, 8], dtype=np.uint8)
    sp_shape = np.array([2, 2], dtype=np.int64)
    dense = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    name = "mul4"
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Example with negative values (int64)
    sp_indices = np.array([[0, 0], [1, 0]], dtype=np.int64)
    sp_values = np.array([-1, 2], dtype=np.int64)
    sp_shape = np.array([2, 2], dtype=np.int64)
    dense = np.array([[3, 4], [5, -6]], dtype=np.int64)
    name = "mul5"
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseDenseCwiseMul"] = tf_raw_ops_sparse_dense_cwise_mul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseDenseCwiseMul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseDenseCwiseMul'.")

check_valid('tf.raw_ops.SparseDenseCwiseMul', generated_inputs['tf.raw_ops.SparseDenseCwiseMul'], lib="tf", suffix=0)
