
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_deserialize_many_sparse_inputs():
    list_of_inputs = []

    def create_sparse_tensor(indices, values, dense_shape, dtype):
        return tf.SparseTensor(indices=indices, values=np.array(values, dtype=dtype), dense_shape=dense_shape)

    # Input 1, valid
    sparse_tensor = create_sparse_tensor([[0, 0], [0, 1], [0, 2]], [1.0, 2.0, 3.0], [1, 3], np.float32)
    serialized_sparse = np.array([[tf.io.serialize_sparse(sparse_tensor).numpy()]], dtype=np.string_)
    dtype = np.float32
    rank = 2
    name = "sparse_tensor_1"

    input_dict = {
        "serialized_sparse": serialized_sparse,
        "dtype": dtype,
        "rank": rank,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    sparse_tensor1 = create_sparse_tensor([[0, 0], [0, 1], [0, 2]], [1, 2, 3], [1, 3], np.int64)
    sparse_tensor2 = create_sparse_tensor([[0, 0], [0, 1], [0, 2]], [4, 5, 6], [1, 3], np.int64)
    serialized_sparse = np.array([[tf.io.serialize_sparse(sparse_tensor1).numpy()], [tf.io.serialize_sparse(sparse_tensor2).numpy()]], dtype=np.string_)
    dtype = np.int64
    rank = 2
    name = "sparse_tensor_2"

    input_dict = {
        "serialized_sparse": serialized_sparse,
        "dtype": dtype,
        "rank": rank,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    sparse_tensor = create_sparse_tensor([[0, 0], [0, 1], [0, 2]], [True, False, True], [1, 3], np.bool_)
    serialized_sparse = np.array([[tf.io.serialize_sparse(sparse_tensor).numpy()]], dtype=np.string_)
    dtype = np.bool_
    rank = 2
    name = "sparse_tensor_3"

    input_dict = {
        "serialized_sparse": serialized_sparse,
        "dtype": dtype,
        "rank": rank,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid, rank = 1
    sparse_tensor = create_sparse_tensor([[0], [1], [2]], [1, 2, 3], [3], np.int32)
    serialized_sparse = np.array([[tf.io.serialize_sparse(sparse_tensor).numpy()]], dtype=np.string_)
    dtype = np.int32
    rank = 1
    name = "sparse_tensor_4"

    input_dict = {
        "serialized_sparse": serialized_sparse,
        "dtype": dtype,
        "rank": rank,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid, 2 sparse tensors, two rows
    sparse_tensor1 = create_sparse_tensor([[0, 0], [0, 1]], [1.0, 2.0], [1, 2], np.float64)
    sparse_tensor2 = create_sparse_tensor([[0, 0]], [3.0], [1, 2], np.float64)
    serialized_sparse = np.array([[tf.io.serialize_sparse(sparse_tensor1).numpy()], [tf.io.serialize_sparse(sparse_tensor2).numpy()]], dtype=np.string_)
    dtype = np.float64
    rank = 2
    name = "sparse_tensor_5"

    input_dict = {
        "serialized_sparse": serialized_sparse,
        "dtype": dtype,
        "rank": rank,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6, valid, different name
    sparse_tensor = create_sparse_tensor([[0, 0], [0, 1], [0, 2]], [1, 2, 3], [1, 3], np.float32)
    serialized_sparse = np.array([[tf.io.serialize_sparse(sparse_tensor).numpy()]], dtype=np.string_)
    dtype = np.float32
    rank = 2
    name = "different_name"

    input_dict = {
        "serialized_sparse": serialized_sparse,
        "dtype": dtype,
        "rank": rank,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid, empty serialized sparse
    serialized_sparse = np.array([[""]*3], dtype=np.string_) # fix for empty
    dtype = np.float32
    rank = 2
    name = "empty_sparse"

    input_dict = {
        "serialized_sparse": serialized_sparse,
        "dtype": dtype,
        "rank": rank,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid, rank = 3
    sparse_tensor = create_sparse_tensor([[0, 0, 0], [0, 0, 1]], [1, 2], [1, 1, 2], np.int32)
    serialized_sparse = np.array([[tf.io.serialize_sparse(sparse_tensor).numpy()]], dtype=np.string_)
    dtype = np.int32
    rank = 3
    name = "sparse_tensor_8"

    input_dict = {
        "serialized_sparse": serialized_sparse,
        "dtype": dtype,
        "rank": rank,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid, multiple sparse tensors, same rank
    sparse_tensor1 = create_sparse_tensor([[0, 0], [0, 1]], [1.0, 2.0], [1, 2], np.float64)
    sparse_tensor2 = create_sparse_tensor([[0, 0]], [3.0], [1, 2], np.float64)
    serialized_sparse = np.array([[tf.io.serialize_sparse(sparse_tensor1).numpy(),tf.io.serialize_sparse(sparse_tensor2).numpy()]], dtype=np.string_)
    dtype = np.float64
    rank = 2
    name = "sparse_tensor_9"

    input_dict = {
        "serialized_sparse": serialized_sparse,
        "dtype": dtype,
        "rank": rank,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    sparse_tensor = create_sparse_tensor([[0, 0], [0, 1], [0, 2]], [1+1j, 2+2j, 3+3j], [1, 3], np.complex64)
    serialized_sparse = np.array([[tf.io.serialize_sparse(sparse_tensor).numpy()]], dtype=np.string_)
    dtype = np.complex64
    rank = 2
    name = "sparse_tensor_10"

    input_dict = {
        "serialized_sparse": serialized_sparse,
        "dtype": dtype,
        "rank": rank,
        "name": name
    }
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
