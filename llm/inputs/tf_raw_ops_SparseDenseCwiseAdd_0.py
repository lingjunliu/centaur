
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
    name = "add_1"
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    sp_indices = np.array([[0, 1], [1, 0], [1, 1]], dtype=np.int64)
    sp_values = np.array([1, 2, 3], dtype=np.int32)
    sp_shape = np.array([2, 2], dtype=np.int64)
    dense = np.array([[4, 5], [6, 7]], dtype=np.int32)
    name = "add_2"
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    sp_indices = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int64)
    sp_values = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    sp_shape = np.array([2, 2, 2], dtype=np.int64)
    dense = np.array([[[0.5, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]], dtype=np.float64)
    name = "add_3"
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    sp_indices = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    sp_values = np.array([-1, -2, -3], dtype=np.int32)
    sp_shape = np.array([3, 3], dtype=np.int64)
    dense = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    name = "add_4"
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    sp_indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    sp_values = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    sp_shape = np.array([2, 2], dtype=np.int64)
    dense = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    name = "add_5"
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    sp_indices = np.array([[0]], dtype=np.int64)
    sp_values = np.array([1], dtype=np.int32)
    sp_shape = np.array([1], dtype=np.int64)
    dense = np.array([2], dtype=np.int32)
    name = "add_6"
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    sp_indices = np.array([[0, 0, 0, 0]], dtype=np.int64)
    sp_values = np.array([1.0], dtype=np.float32)
    sp_shape = np.array([1,1,1,1], dtype=np.int64)
    dense = np.array([[[[2.0]]]], dtype=np.float32)
    name = "add_7"
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    sp_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    sp_values = np.array([1, 2], dtype=np.int64)
    sp_shape = np.array([2, 2], dtype=np.int64)
    dense = np.array([[3, 4], [5, 6]], dtype=np.int64)
    name = "add_8"
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    sp_indices = np.array([[0, 0, 0]], dtype=np.int64)
    sp_values = np.array([1], dtype=np.int64)
    sp_shape = np.array([1, 1, 1], dtype=np.int64)
    dense = np.array([[[2]]], dtype=np.int64)
    name = "add_9"
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    sp_indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    sp_values = np.array([1.0, 2.0], dtype=np.float64)
    sp_shape = np.array([2, 2, 2], dtype=np.int64)
    dense = np.array([[[3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0]]], dtype=np.float64)
    name = "add_10"
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "dense": dense, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_SparseDenseCwiseAdd_inputs()
generated_inputs["tf.raw_ops.SparseDenseCwiseAdd"] = []
for input_dict in inputs:
    kwargs = {}
    for key, value in input_dict.items():
        kwargs[key] = value
    generated_inputs["tf.raw_ops.SparseDenseCwiseAdd"].append({"kwargs": kwargs})

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseDenseCwiseAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseDenseCwiseAdd'.")

check_valid('tf.raw_ops.SparseDenseCwiseAdd', generated_inputs['tf.raw_ops.SparseDenseCwiseAdd'], lib="tf", suffix=0)
