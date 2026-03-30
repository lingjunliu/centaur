
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sparse_split_inputs():
    list_of_inputs = []

    # Input 1
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    values = np.array([1, 2, 3, 4], dtype=np.int32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    num_split = 2
    axis = np.array(1, dtype=np.int64)
    name = "split_1"
    input_dict = {"sp_input": sp_input, "num_split": num_split, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], dtype=np.int64)
    values = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    num_split = 3
    axis = np.array(1, dtype=np.int64)
    name = "split_2"
    input_dict = {"sp_input": sp_input, "num_split": num_split, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3 - Splitting along axis 0
    indices = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    dense_shape = np.array([3, 3], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    num_split = 3
    axis = np.array(0, dtype=np.int64)
    name = "split_3"
    input_dict = {"sp_input": sp_input, "num_split": num_split, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 - num_split > dimension size, should still work and produce empty tensors
    indices = np.array([[0, 0]], dtype=np.int64)
    values = np.array([1], dtype=np.int32)
    dense_shape = np.array([1, 1], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    num_split = 1
    axis = np.array(1, dtype=np.int64)
    name = "split_4"
    input_dict = {"sp_input": sp_input, "num_split": num_split, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 - Negative axis
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    num_split = 2
    axis = np.array(-1, dtype=np.int64)
    name = "split_5"
    input_dict = {"sp_input": sp_input, "num_split": num_split, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 - Different dtype for values
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    values = np.array([1.0, 2.0], dtype=np.float32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    num_split = 2
    axis = np.array(1, dtype=np.int64)
    name = "split_6"
    input_dict = {"sp_input": sp_input, "num_split": num_split, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - 3D SparseTensor
    indices = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    dense_shape = np.array([2, 2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    num_split = 2
    axis = np.array(2, dtype=np.int64)
    name = "split_7"
    input_dict = {"sp_input": sp_input, "num_split": num_split, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 - uneven split
    indices = np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=np.int64)
    values = np.array([1, 2, 3, 4], dtype=np.int32)
    dense_shape = np.array([1, 4], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    num_split = 3
    axis = np.array(1, dtype=np.int64)
    name = "split_8"
    input_dict = {"sp_input": sp_input, "num_split": num_split, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - different indices dtype
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    num_split = 2
    axis = np.array(1, dtype=np.int64)
    name = "split_9"
    input_dict = {"sp_input": sp_input, "num_split": num_split, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - empty SparseTensor
    indices = np.empty((0, 2), dtype=np.int64)
    values = np.empty((0,), dtype=np.int32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    num_split = 2
    axis = np.array(1, dtype=np.int64)
    name = "split_10"
    input_dict = {"sp_input": sp_input, "num_split": num_split, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.split"] = tf_sparse_split_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.split' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.split'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sparse.split', generated_inputs['tf.sparse.split'], lib="tf", suffix=0)
