
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_deserialize_many_sparse_inputs():
    list_of_inputs = []

    # Input 1
    indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int64)
    shape = np.array([1, 2], dtype=np.int64)
    sparse_tensor = tf.SparseTensor(indices, values, shape)
    serialized_sparse = tf.io.serialize_sparse(sparse_tensor)
    serialized_sparse = np.array([[serialized_sparse.numpy()]], dtype=object)

    dtype = np.int64
    rank = 2
    name = "sparse_tensor_1"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices1 = np.array([[0, 0], [0, 1]], dtype=np.int64)
    values1 = np.array([1.0, 2.0], dtype=np.float32)
    shape1 = np.array([1, 2], dtype=np.int64)
    sparse_tensor1 = tf.SparseTensor(indices1, values1, shape1)
    serialized_sparse1 = tf.io.serialize_sparse(sparse_tensor1)

    indices2 = np.array([[0, 0], [0, 1]], dtype=np.int64)
    values2 = np.array([4.0, 5.0], dtype=np.float32)
    shape2 = np.array([1, 2], dtype=np.int64)
    sparse_tensor2 = tf.SparseTensor(indices2, values2, shape2)
    serialized_sparse2 = tf.io.serialize_sparse(sparse_tensor2)

    serialized_sparse = np.array([[serialized_sparse1.numpy()], [serialized_sparse2.numpy()]], dtype=object)

    dtype = np.float32
    rank = 2
    name = "sparse_tensor_2"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int32)
    shape = np.array([1, 2], dtype=np.int64)
    sparse_tensor = tf.SparseTensor(indices, values, shape)
    serialized_sparse = tf.io.serialize_sparse(sparse_tensor)
    serialized_sparse = np.array([[serialized_sparse.numpy()]], dtype=object)

    dtype = np.int32
    rank = 2
    name = None
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    values = np.array([1.0, 2.0], dtype=np.float64)
    shape = np.array([1, 2], dtype=np.int64)
    sparse_tensor = tf.SparseTensor(indices, values, shape)
    serialized_sparse = tf.io.serialize_sparse(sparse_tensor)
    serialized_sparse = np.array([[serialized_sparse.numpy()]], dtype=object)

    dtype = np.float64
    rank = 2
    name = "sparse_tensor_4"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    indices1 = np.array([[0, 0], [0, 1]], dtype=np.int64)
    values1 = np.array([True, False], dtype=np.bool_)
    shape1 = np.array([1, 2], dtype=np.int64)
    sparse_tensor1 = tf.SparseTensor(indices1, values1, shape1)
    serialized_sparse1 = tf.io.serialize_sparse(sparse_tensor1)

    indices2 = np.array([[0, 0], [0, 1]], dtype=np.int64)
    values2 = np.array([True, True], dtype=np.bool_)
    shape2 = np.array([1, 2], dtype=np.int64)
    sparse_tensor2 = tf.SparseTensor(indices2, values2, shape2)
    serialized_sparse2 = tf.io.serialize_sparse(sparse_tensor2)
    serialized_sparse = np.array([[serialized_sparse1.numpy()], [serialized_sparse2.numpy()]], dtype=object)

    dtype = np.bool_
    rank = 2
    name = "sparse_tensor_5"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int16)
    shape = np.array([1, 2], dtype=np.int64)
    sparse_tensor = tf.SparseTensor(indices, values, shape)
    serialized_sparse = tf.io.serialize_sparse(sparse_tensor)
    serialized_sparse = np.array([[serialized_sparse.numpy()]], dtype=object)

    dtype = np.int16
    rank = 2
    name = "sparse_tensor_6"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int8)
    shape = np.array([1, 2], dtype=np.int64)
    sparse_tensor = tf.SparseTensor(indices, values, shape)
    serialized_sparse = tf.io.serialize_sparse(sparse_tensor)
    serialized_sparse = np.array([[serialized_sparse.numpy()]], dtype=object)

    dtype = np.int8
    rank = 2
    name = "sparse_tensor_7"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.uint8)
    shape = np.array([1, 2], dtype=np.int64)
    sparse_tensor = tf.SparseTensor(indices, values, shape)
    serialized_sparse = tf.io.serialize_sparse(sparse_tensor)
    serialized_sparse = np.array([[serialized_sparse.numpy()]], dtype=object)

    dtype = np.uint8
    rank = 2
    name = "sparse_tensor_8"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    indices1 = np.array([[0, 0], [0, 1]], dtype=np.int64)
    values1 = np.array([1, 2], dtype=np.uint16)
    shape1 = np.array([1, 2], dtype=np.int64)
    sparse_tensor1 = tf.SparseTensor(indices1, values1, shape1)
    serialized_sparse1 = tf.io.serialize_sparse(sparse_tensor1)

    indices2 = np.array([[0, 0], [0, 1]], dtype=np.int64)
    values2 = np.array([4, 5], dtype=np.uint16)
    shape2 = np.array([1, 2], dtype=np.int64)
    sparse_tensor2 = tf.SparseTensor(indices2, values2, shape2)
    serialized_sparse2 = tf.io.serialize_sparse(sparse_tensor2)
    serialized_sparse = np.array([[serialized_sparse1.numpy()], [serialized_sparse2.numpy()]], dtype=object)

    dtype = np.uint16
    rank = 2
    name = "sparse_tensor_9"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    values = np.array([1+1j, 2+2j], dtype=np.complex64)
    shape = np.array([1, 2], dtype=np.int64)
    sparse_tensor = tf.SparseTensor(indices, values, shape)
    serialized_sparse = tf.io.serialize_sparse(sparse_tensor)
    serialized_sparse = np.array([[serialized_sparse.numpy()]], dtype=object)

    dtype = np.complex64
    rank = 2
    name = "sparse_tensor_10"
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
