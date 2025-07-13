
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
    name = "sparse_softmax_1"
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    sp_indices = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    sp_values = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    sp_shape = np.array([3, 3], dtype=np.int64)
    name = "sparse_softmax_2"
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    sp_indices = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=np.int64)
    sp_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    sp_shape = np.array([2, 2, 2], dtype=np.int64)
    name = "sparse_softmax_3"
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    sp_indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    sp_values = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32)
    sp_shape = np.array([2, 2], dtype=np.int64)
    name = "sparse_softmax_4"
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    sp_indices = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=np.int64)
    sp_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    sp_shape = np.array([2, 2, 2], dtype=np.int64)
    name = "sparse_softmax_5"
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    sp_indices = np.array([[0, 0], [1, 0], [1, 1], [2, 1], [2, 2]], dtype=np.int64)
    sp_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    sp_shape = np.array([3, 3], dtype=np.int64)
    name = "sparse_softmax_6"
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    sp_indices = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=np.int64)
    sp_values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    sp_shape = np.array([2, 2, 2], dtype=np.int64)
    name = "sparse_softmax_7"
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: half type
    sp_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    sp_values = np.array([1.0, 2.0], dtype=np.float16)
    sp_shape = np.array([2, 2], dtype=np.int64)
    name = "sparse_softmax_8"
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: larger shape
    sp_indices = np.array([[0, 0], [1, 1], [2, 2], [3,3]], dtype=np.int64)
    sp_values = np.array([0.1, 0.5, 0.9, 0.2], dtype=np.float64)
    sp_shape = np.array([4, 4], dtype=np.int64)
    name = "sparse_softmax_9"
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: bigger indices
    sp_indices = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 1], [2,1,1]], dtype=np.int64)
    sp_values = np.array([1.0, 2.0, 3.0, 1.0], dtype=np.float32)
    sp_shape = np.array([3, 2, 2], dtype=np.int64)
    name = "sparse_softmax_10"
    input_dict = {"sp_indices": sp_indices, "sp_values": sp_values, "sp_shape": sp_shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseSoftmax"] = []
for input_data in tf_raw_ops_SparseSoftmax_inputs():
  generated_inputs["tf.raw_ops.SparseSoftmax"].append({"args": (), "kwargs": input_data})

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseSoftmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseSoftmax'.")

check_valid('tf.raw_ops.SparseSoftmax', generated_inputs['tf.raw_ops.SparseSoftmax'], lib="tf", suffix=0)
