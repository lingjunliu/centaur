
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_serialize_sparse_inputs():
    list_of_inputs = []

    # Input 1
    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.float32)
    dense_shape = np.array([3, 4], dtype=np.int64)
    sp_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    out_type = tf.string
    name = "sparse_tensor_1"
    input_dict = {"sp_input": sp_input, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = np.array([[0, 1], [1, 0], [2, 2]], dtype=np.int64)
    values = np.array([3, 4, 5], dtype=np.int32)
    dense_shape = np.array([3, 3], dtype=np.int64)
    sp_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    out_type = tf.string
    name = "sparse_tensor_2"
    input_dict = {"sp_input": sp_input, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int64)
    values = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    dense_shape = np.array([2, 2, 2], dtype=np.int64)
    sp_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    out_type = tf.string
    name = "sparse_tensor_3"
    input_dict = {"sp_input": sp_input, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    indices = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    values = np.array([-1, -2, -3], dtype=np.int32)
    dense_shape = np.array([3, 3], dtype=np.int64)
    sp_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    out_type = tf.string
    name = "sparse_tensor_4"
    input_dict = {"sp_input": sp_input, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    indices = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int64)
    values = np.array([True, False], dtype=np.bool_)
    dense_shape = np.array([1, 1, 2], dtype=np.int64)
    sp_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    out_type = tf.string
    name = "sparse_tensor_5"
    input_dict = {"sp_input": sp_input, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.float32)
    dense_shape = np.array([3, 4], dtype=np.int64)
    sp_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    out_type = tf.string
    name = None
    input_dict = {"sp_input": sp_input, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    indices = np.array([[0, 1], [1, 0], [2, 2]], dtype=np.int64)
    values = np.array([3, 4, 5], dtype=np.int32)
    dense_shape = np.array([3, 3], dtype=np.int64)
    sp_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    out_type = tf.string
    name = ""
    input_dict = {"sp_input": sp_input, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    indices = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int64)
    values = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    dense_shape = np.array([2, 2, 2], dtype=np.int64)
    sp_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    out_type = tf.string
    name = " "
    input_dict = {"sp_input": sp_input, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    indices = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    values = np.array([-1, -2, -3], dtype=np.int32)
    dense_shape = np.array([3, 3], dtype=np.int64)
    sp_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    out_type = tf.string
    name = "  "
    input_dict = {"sp_input": sp_input, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10
    indices = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int64)
    values = np.array([True, False], dtype=np.bool_)
    dense_shape = np.array([1, 1, 2], dtype=np.int64)
    sp_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    out_type = tf.string
    name = "name"
    input_dict = {"sp_input": sp_input, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.serialize_sparse"] = tf_io_serialize_sparse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.serialize_sparse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.serialize_sparse'.")

check_valid('tf.io.serialize_sparse', generated_inputs['tf.io.serialize_sparse'], lib="tf", suffix=0)
