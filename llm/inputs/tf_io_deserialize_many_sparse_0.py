
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_deserialize_many_sparse_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    values = np.array([1, 2, 3, 4], dtype=np.float32)
    shape = np.array([2, 2], dtype=np.int64)
    st = tf.SparseTensor(indices, values, shape)
    serialized_sparse = tf.io.serialize_sparse(st)
    serialized_sparse = np.array([[serialized_sparse.numpy()], [serialized_sparse.numpy()]], dtype=object)

    dtype = np.float32
    rank = 2
    name = "sparse_tensor_1"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different dtype
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    values = np.array([1, 2, 3, 4], dtype=np.int32)
    shape = np.array([2, 2], dtype=np.int64)
    st = tf.SparseTensor(indices, values, shape)
    serialized_sparse = tf.io.serialize_sparse(st)
    serialized_sparse = np.array([[serialized_sparse.numpy()], [serialized_sparse.numpy()]], dtype=object)
    dtype = np.int32
    rank = 2
    name = "sparse_tensor_2"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different rank
    indices = np.array([[0], [1], [2]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.float32)
    shape = np.array([3], dtype=np.int64)
    st = tf.SparseTensor(indices, values, shape)
    serialized_sparse = tf.io.serialize_sparse(st)
    serialized_sparse = np.array([[serialized_sparse.numpy()], [serialized_sparse.numpy()]], dtype=object)
    dtype = np.float32
    rank = 1
    name = "sparse_tensor_3"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different minibatch size (N=3)
    indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.float32)
    shape = np.array([1, 2], dtype=np.int64)
    st = tf.SparseTensor(indices, values, shape)
    serialized_sparse = tf.io.serialize_sparse(st)
    serialized_sparse = np.array([[serialized_sparse.numpy()], [serialized_sparse.numpy()], [serialized_sparse.numpy()]], dtype=object)

    dtype = np.float32
    rank = 2
    name = "sparse_tensor_4"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different name
    indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.float32)
    shape = np.array([1, 2], dtype=np.int64)
    st = tf.SparseTensor(indices, values, shape)
    serialized_sparse = tf.io.serialize_sparse(st)
    serialized_sparse = np.array([[serialized_sparse.numpy()], [serialized_sparse.numpy()]], dtype=object)
    dtype = np.float32
    rank = 2
    name = "another_sparse_tensor"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: dtype=tf.string
    indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    values = np.array(["a", "b"], dtype=np.string_)
    shape = np.array([1, 2], dtype=np.int64)
    st = tf.SparseTensor(indices, values, shape)
    serialized_sparse = tf.io.serialize_sparse(st)
    serialized_sparse = np.array([[serialized_sparse.numpy()], [serialized_sparse.numpy()]], dtype=object)
    dtype = np.string_
    rank = 2
    name = "string_tensor"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different shapes in minibatch. Pad to the max shape.
    indices1 = np.array([[0, 0], [0, 1]], dtype=np.int64)
    values1 = np.array([1, 2], dtype=np.float32)
    shape1 = np.array([1, 2], dtype=np.int64)
    st1 = tf.SparseTensor(indices1, values1, shape1)
    serialized_sparse1 = tf.io.serialize_sparse(st1)

    indices2 = np.array([[0, 0], [1, 1]], dtype=np.int64)
    values2 = np.array([3, 4], dtype=np.float32)
    shape2 = np.array([2, 2], dtype=np.int64)
    st2 = tf.SparseTensor(indices2, values2, shape2)
    serialized_sparse2 = tf.io.serialize_sparse(st2)

    serialized_sparse = np.array([[serialized_sparse1.numpy()], [serialized_sparse2.numpy()]], dtype=object)
    dtype = np.float32
    rank = 2
    name = "diff_dims"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.deserialize_many_sparse"] = tf_io_deserialize_many_sparse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.deserialize_many_sparse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.deserialize_many_sparse'.")

check_valid('tf.io.deserialize_many_sparse', generated_inputs['tf.io.deserialize_many_sparse'], lib="tf", suffix=0)
