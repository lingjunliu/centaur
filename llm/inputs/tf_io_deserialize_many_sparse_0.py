
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_deserialize_many_sparse_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    indices = np.array([[0], [1], [2]], dtype=np.int64)
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shape = np.array([5], dtype=np.int64)
    sparse_tensor = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    serialized_sparse = tf.io.serialize_sparse(sparse_tensor)
    serialized_sparse_numpy = serialized_sparse.numpy()
    serialized_sparse = np.array([serialized_sparse_numpy, serialized_sparse_numpy], dtype=np.object_)
    serialized_sparse = np.stack([serialized_sparse, serialized_sparse, serialized_sparse], axis = 1)

    dtype = np.float32
    rank = 1
    name = "sparse_tensor_1"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different dtype
    indices = np.array([[0], [1], [2]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    shape = np.array([5], dtype=np.int64)
    sparse_tensor = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    serialized_sparse = tf.io.serialize_sparse(sparse_tensor)
    serialized_sparse_numpy = serialized_sparse.numpy()
    serialized_sparse = np.array([serialized_sparse_numpy, serialized_sparse_numpy], dtype=np.object_)
    serialized_sparse = np.stack([serialized_sparse, serialized_sparse, serialized_sparse], axis = 1)

    dtype = np.int32
    rank = 1
    name = "sparse_tensor_2"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different rank
    indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shape = np.array([2, 2], dtype=np.int64)
    sparse_tensor = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    serialized_sparse = tf.io.serialize_sparse(sparse_tensor)
    serialized_sparse_numpy = serialized_sparse.numpy()
    serialized_sparse = np.array([serialized_sparse_numpy, serialized_sparse_numpy], dtype=np.object_)
    serialized_sparse = np.stack([serialized_sparse, serialized_sparse, serialized_sparse], axis = 1)

    dtype = np.float32
    rank = 2
    name = "sparse_tensor_3"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty name
    indices = np.array([[0], [1], [2]], dtype=np.int64)
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shape = np.array([5], dtype=np.int64)
    sparse_tensor = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    serialized_sparse = tf.io.serialize_sparse(sparse_tensor)
    serialized_sparse_numpy = serialized_sparse.numpy()
    serialized_sparse = np.array([serialized_sparse_numpy, serialized_sparse_numpy], dtype=np.object_)
    serialized_sparse = np.stack([serialized_sparse, serialized_sparse, serialized_sparse], axis = 1)

    dtype = np.float32
    rank = 1
    name = ""
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Rank as None
    indices = np.array([[0], [1], [2]], dtype=np.int64)
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shape = np.array([5], dtype=np.int64)
    sparse_tensor = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    serialized_sparse = tf.io.serialize_sparse(sparse_tensor)
    serialized_sparse_numpy = serialized_sparse.numpy()
    serialized_sparse = np.array([serialized_sparse_numpy, serialized_sparse_numpy], dtype=np.object_)
    serialized_sparse = np.stack([serialized_sparse, serialized_sparse, serialized_sparse], axis = 1)

    dtype = np.float32
    rank = None
    name = "sparse_tensor_5"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6: minibatch size 1
    indices = np.array([[0], [1], [2]], dtype=np.int64)
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shape = np.array([5], dtype=np.int64)
    sparse_tensor = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    serialized_sparse = tf.io.serialize_sparse(sparse_tensor)
    serialized_sparse_numpy = serialized_sparse.numpy()
    serialized_sparse = np.array([serialized_sparse_numpy], dtype=np.object_)
    serialized_sparse = np.stack([serialized_sparse, serialized_sparse, serialized_sparse], axis = 1)


    dtype = np.float32
    rank = 1
    name = "sparse_tensor_6"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different name
    indices = np.array([[0], [1], [2]], dtype=np.int64)
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shape = np.array([5], dtype=np.int64)
    sparse_tensor = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    serialized_sparse = tf.io.serialize_sparse(sparse_tensor)
    serialized_sparse_numpy = serialized_sparse.numpy()
    serialized_sparse = np.array([serialized_sparse_numpy, serialized_sparse_numpy], dtype=np.object_)
    serialized_sparse = np.stack([serialized_sparse, serialized_sparse, serialized_sparse], axis = 1)

    dtype = np.float32
    rank = 1
    name = "different_name"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Rank 3
    indices = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.int64)
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shape = np.array([1, 2, 2], dtype=np.int64)
    sparse_tensor = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    serialized_sparse = tf.io.serialize_sparse(sparse_tensor)
    serialized_sparse_numpy = serialized_sparse.numpy()
    serialized_sparse = np.array([serialized_sparse_numpy, serialized_sparse_numpy], dtype=np.object_)
    serialized_sparse = np.stack([serialized_sparse, serialized_sparse, serialized_sparse], axis = 1)

    dtype = np.float32
    rank = 3
    name = "sparse_tensor_8"
    input_dict = {"serialized_sparse": serialized_sparse, "dtype": dtype, "rank": rank, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Int64
    indices = np.array([[0], [1], [2]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int64)
    shape = np.array([5], dtype=np.int64)
    sparse_tensor = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    serialized_sparse = tf.io.serialize_sparse(sparse_tensor)
    serialized_sparse_numpy = serialized_sparse.numpy()
    serialized_sparse = np.array([serialized_sparse_numpy, serialized_sparse_numpy], dtype=np.object_)
    serialized_sparse = np.stack([serialized_sparse, serialized_sparse, serialized_sparse], axis = 1)

    dtype = np.int64
    rank = 1
    name = "sparse_tensor_9"
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
