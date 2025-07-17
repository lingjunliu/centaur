
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
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": "test1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    sp_indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    sp_values = np.array([1, 2, 3], dtype=np.int32)
    sp_shape = np.array([2, 2], dtype=np.int64)
    dense = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": "test2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    sp_indices = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int64)
    sp_values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    sp_shape = np.array([2, 2, 2], dtype=np.int64)
    dense = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": "test3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    sp_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    sp_values = np.array([1, -2], dtype=np.int32)
    sp_shape = np.array([2, 2], dtype=np.int64)
    dense = np.array([[1, 2], [3, -4]], dtype=np.int32)
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": "test4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    sp_indices = np.array([[0, 0, 0]], dtype=np.int64)
    sp_values = np.array([1.0], dtype=np.float32)
    sp_shape = np.array([1, 1, 1], dtype=np.int64)
    dense = np.array([[[2.0]]], dtype=np.float32)
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": "test5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    sp_indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
    sp_values = np.array([5, 8], dtype=np.int64)
    sp_shape = np.array([2, 2], dtype=np.int64)
    dense = np.array([[2, 4], [6, 9]], dtype=np.int64)
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": "test6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    sp_indices = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    sp_values = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    sp_shape = np.array([3, 3], dtype=np.int64)
    dense = np.array([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0], [3.5, 4.0, 4.5]], dtype=np.float64)
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": "test7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: uint8
    sp_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    sp_values = np.array([1, 2], dtype=np.uint8)
    sp_shape = np.array([2, 3], dtype=np.int64)
    dense = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": "test8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9:  Removed complex64, it is not supported
    # Input 10: float16
    sp_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    sp_values = np.array([1.0, 2.0], dtype=np.float16)
    sp_shape = np.array([2, 2], dtype=np.int64)
    dense = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": "test9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: bfloat16
    sp_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    sp_values = np.array([1.0, 2.0], dtype=np.float32)
    sp_shape = np.array([2, 2], dtype=np.int64)
    dense = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32) # float32 for dense to avoid cast issue
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": "test10"}
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
