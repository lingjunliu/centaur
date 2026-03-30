
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sparse_reshape_inputs():
    list_of_inputs = []

    # Input 1
    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int32)
    shape = np.array([3, 4], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([4, 3], dtype=np.int64)
    input_dict = {"sp_input": sp_input, "shape": new_shape, "name": "reshape_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    shape = np.array([2, 2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([2, 4], dtype=np.int64)
    input_dict = {"sp_input": sp_input, "shape": new_shape, "name": "reshape_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    values = np.array([4, 5, 6], dtype=np.int32)
    shape = np.array([3, 3], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([9], dtype=np.int64)
    input_dict = {"sp_input": sp_input, "shape": new_shape, "name": "reshape_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    values = np.array([7, 8, 9], dtype=np.int32)
    shape = np.array([2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([1, 4], dtype=np.int64)
    input_dict = {"sp_input": sp_input, "shape": new_shape, "name": "reshape_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    indices = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int64)
    values = np.array([10, 11], dtype=np.int32)
    shape = np.array([1, 1, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([2], dtype=np.int64)
    input_dict = {"sp_input": sp_input, "shape": new_shape, "name": "reshape_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int32)
    shape = np.array([2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([4, ], dtype=np.int64)
    input_dict = {"sp_input": sp_input, "shape": new_shape, "name": "reshape_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Using -1
    indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int32)
    shape = np.array([2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([1, -1], dtype=np.int64)
    input_dict = {"sp_input": sp_input, "shape": new_shape, "name": "reshape_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Using -1 with different data
    indices = np.array([[0, 0, 0], [0, 1, 0]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int32)
    shape = np.array([1, 2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([4, -1], dtype=np.int64)
    input_dict = {"sp_input": sp_input, "shape": new_shape, "name": "reshape_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    indices = np.array([[0, 0], [1, 1], [2, 0]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    shape = np.array([3, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([2,3], dtype=np.int64)
    input_dict = {"sp_input": sp_input, "shape": new_shape, "name": "reshape_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    indices = np.array([[0, 0, 0], [1, 0, 1], [1, 1, 0]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    shape = np.array([2, 2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([4,-1], dtype=np.int64)
    input_dict = {"sp_input": sp_input, "shape": new_shape, "name": "reshape_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.reshape"] = tf_sparse_reshape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.reshape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.reshape'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sparse.reshape', generated_inputs['tf.sparse.reshape'], lib="tf", suffix=0)
