
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseSoftmax_inputs():
    list_of_inputs = []

    # Input 1
    sp_indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    sp_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    sp_shape = np.array([2, 2], dtype=np.int64)
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "name": "sparse_softmax_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    sp_indices = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.int64)
    sp_values = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    sp_shape = np.array([1, 2, 2], dtype=np.int64)
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "name": "sparse_softmax_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    sp_indices = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    sp_values = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    sp_shape = np.array([3, 3], dtype=np.int64)
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "name": "sparse_softmax_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    sp_indices = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int64)
    sp_values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    sp_shape = np.array([2, 2, 2], dtype=np.int64)
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "name": "sparse_softmax_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    sp_indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0]], dtype=np.int64)
    sp_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    sp_shape = np.array([3, 2], dtype=np.int64)
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "name": "sparse_softmax_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    sp_indices = np.array([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=np.int64)
    sp_values = np.array([1.0, 2.0], dtype=np.float64)
    sp_shape = np.array([1, 1, 1, 2], dtype=np.int64)
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "name": "sparse_softmax_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    sp_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    sp_values = np.array([1.0, -1.0], dtype=np.float32)
    sp_shape = np.array([2, 2], dtype=np.int64)
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "name": "sparse_softmax_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    sp_indices = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int64)
    sp_values = np.array([-2.0, -3.0], dtype=np.float64)
    sp_shape = np.array([1, 1, 2], dtype=np.int64)
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "name": "sparse_softmax_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    sp_indices = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int64)
    sp_values = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    sp_shape = np.array([2, 2], dtype=np.int64)
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "name": "sparse_softmax_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    sp_indices = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1]], dtype=np.int64)
    sp_values = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    sp_shape = np.array([1, 2, 2], dtype=np.int64)
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "name": "sparse_softmax_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseSoftmax"] = tf_raw_ops_SparseSoftmax_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseSoftmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseSoftmax'.")

check_valid('tf.raw_ops.SparseSoftmax', generated_inputs['tf.raw_ops.SparseSoftmax'], lib="tf", suffix=0)
