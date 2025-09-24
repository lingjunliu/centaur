
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseReorder_inputs():
    list_of_inputs = []

    # Input 1
    input_indices = np.array([[0, 1], [0, 0], [1, 0]], dtype=np.int64)
    input_values = np.array([5, 2, 7], dtype=np.int32)
    input_shape = np.array([2, 2], dtype=np.int64)
    input_dict = {"input_indices": input_indices, "input_values": input_values, "input_shape": input_shape, "name": "reorder1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    input_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_shape = np.array([2, 2], dtype=np.int64)
    input_dict = {"input_indices": input_indices, "input_values": input_values, "input_shape": input_shape, "name": "reorder2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3 (3D Tensor)
    input_indices = np.array([[0, 0, 1], [0, 0, 0], [0, 1, 0]], dtype=np.int64)
    input_values = np.array([4, 1, 3], dtype=np.int32)
    input_shape = np.array([1, 2, 2], dtype=np.int64)
    input_dict = {"input_indices": input_indices, "input_values": input_values, "input_shape": input_shape, "name": "reorder3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 (Larger Tensor)
    input_indices = np.array([[0, 0], [1, 1], [2, 0], [2, 1]], dtype=np.int64)
    input_values = np.array([10, 20, 30, 40], dtype=np.int32)
    input_shape = np.array([3, 2], dtype=np.int64)
    input_dict = {"input_indices": input_indices, "input_values": input_values, "input_shape": input_shape, "name": "reorder4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (Unsorted indices with same row)
    input_indices = np.array([[0, 1], [0, 0], [1, 0], [1,1]], dtype=np.int64)
    input_values = np.array([5, 2, 7, 9], dtype=np.int32)
    input_shape = np.array([2, 2], dtype=np.int64)
    input_dict = {"input_indices": input_indices, "input_values": input_values, "input_shape": input_shape, "name": "reorder5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (float64 values)
    input_indices = np.array([[0, 1], [0, 0], [1, 0]], dtype=np.int64)
    input_values = np.array([5.0, 2.0, 7.0], dtype=np.float64)
    input_shape = np.array([2, 2], dtype=np.int64)
    input_dict = {"input_indices": input_indices, "input_values": input_values, "input_shape": input_shape, "name": "reorder6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (int64 values)
    input_indices = np.array([[0, 1], [0, 0], [1, 0]], dtype=np.int64)
    input_values = np.array([5, 2, 7], dtype=np.int64)
    input_shape = np.array([2, 2], dtype=np.int64)
    input_dict = {"input_indices": input_indices, "input_values": input_values, "input_shape": input_shape, "name": "reorder7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (shape with higher values)
    input_indices = np.array([[0, 1], [0, 0], [1, 0]], dtype=np.int64)
    input_values = np.array([5, 2, 7], dtype=np.int32)
    input_shape = np.array([100, 100], dtype=np.int64)
    input_dict = {"input_indices": input_indices, "input_values": input_values, "input_shape": input_shape, "name": "reorder8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (4D Tensor)
    input_indices = np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 1, 0]], dtype=np.int64)
    input_values = np.array([4, 1, 3], dtype=np.int32)
    input_shape = np.array([1, 1, 2, 2], dtype=np.int64)
    input_dict = {"input_indices": input_indices, "input_values": input_values, "input_shape": input_shape, "name": "reorder9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (empty values, but valid shape)
    input_indices = np.array([], dtype=np.int64).reshape(0,2)
    input_values = np.array([], dtype=np.int32)
    input_shape = np.array([2, 2], dtype=np.int64)
    input_dict = {"input_indices": input_indices, "input_values": input_values, "input_shape": input_shape, "name": "reorder10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseReorder"] = tf_raw_ops_SparseReorder_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseReorder' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseReorder'.")

check_valid('tf.raw_ops.SparseReorder', generated_inputs['tf.raw_ops.SparseReorder'], lib="tf", suffix=0)
