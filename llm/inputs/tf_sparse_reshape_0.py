
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_reshape_inputs():
    list_of_inputs = []

    def get_tensor_size(tensor):
        if isinstance(tensor, tf.SparseTensor):
            return np.prod(tensor.dense_shape.numpy())
        else:
            return tensor.size

    # Input 1: Basic valid case
    indices = np.array([[0, 0], [1, 2], [2, 0]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    shape = np.array([3, 3], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    new_shape = np.array([9], dtype=np.int64)
    input_dict = {"sp_input": sp_input, "shape": new_shape, "name": "reshape_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Using -1 for shape inference
    indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    shape = np.array([2, 2], dtype=np.int64) #originally 4 elements
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    new_shape = np.array([4, -1], dtype=np.int64) #should become 4, 1
    input_dict = {"sp_input": sp_input, "shape": new_shape, "name": "reshape_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Reshaping to a different dimension
    indices = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    shape = np.array([2, 2, 2], dtype=np.int64) #8
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    new_shape = np.array([4, 2], dtype=np.int64)
    input_dict = {"sp_input": sp_input, "shape": new_shape, "name": "reshape_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty SparseTensor
    indices = np.array([], dtype=np.int64).reshape(0, 2)
    values = np.array([], dtype=np.int32)
    shape = np.array([2, 3], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    new_shape = np.array([6], dtype=np.int64)
    input_dict = {"sp_input": sp_input, "shape": new_shape, "name": "reshape_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Reshape into a scalar
    indices = np.array([[0, 0]], dtype=np.int64)
    values = np.array([5], dtype=np.int32)
    shape = np.array([1, 1], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    new_shape = np.array([1], dtype=np.int64)
    input_dict = {"sp_input": sp_input, "shape": new_shape, "name": "reshape_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Larger sparse tensor
    indices = np.array([[0, 0], [1, 1], [2, 2], [3,3], [4,4]], dtype=np.int64)
    values = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    shape = np.array([5, 5], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    new_shape = np.array([25], dtype=np.int64)
    input_dict = {"sp_input": sp_input, "shape": new_shape, "name": "reshape_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Reshape to a single row
    indices = np.array([[0, 0], [1, 1], [2, 0]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    shape = np.array([3, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    new_shape = np.array([1, 6], dtype=np.int64)
    input_dict = {"sp_input": sp_input, "shape": new_shape, "name": "reshape_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different data type for values
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    values = np.array([1.0, 2.5], dtype=np.float32)
    shape = np.array([2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    new_shape = np.array([4], dtype=np.int64)
    input_dict = {"sp_input": sp_input, "shape": new_shape, "name": "reshape_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Higher rank tensor
    indices = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int32)
    shape = np.array([2, 2, 2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    new_shape = np.array([16], dtype=np.int64)
    input_dict = {"sp_input": sp_input, "shape": new_shape, "name": "reshape_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Reshaping with -1 and larger tensor
    indices = np.array([[0, 0], [0, 1], [1, 0], [2, 2], [3, 1]], dtype=np.int64)
    values = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    shape = np.array([4, 3], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    new_shape = np.array([6, -1], dtype=np.int64)
    input_dict = {"sp_input": sp_input, "shape": new_shape, "name": "reshape_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.reshape"] = tf_sparse_reshape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.reshape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.reshape'.")

check_valid('tf.sparse.reshape', generated_inputs['tf.sparse.reshape'], lib="tf", suffix=0)
