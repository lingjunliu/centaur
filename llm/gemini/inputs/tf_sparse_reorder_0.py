
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

def tf_sparse_reorder_inputs():
    list_of_inputs = []

    # Input 1
    indices = np.array([[0, 3], [0, 1], [3, 1], [2, 0]], dtype=np.int64)
    values = np.array([1, 2, 3, 4], dtype=np.int32)
    shape = np.array([4, 5], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_tensor_1"}
    list_of_inputs.append(input_dict)

    # Input 2
    indices = np.array([[0, 0], [1, 2], [1, 1], [0, 2]], dtype=np.int64)
    values = np.array([1, 2, 3, 4], dtype=np.int32)
    shape = np.array([2, 3], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_tensor_2"}
    list_of_inputs.append(input_dict)

    # Input 3 (3D SparseTensor)
    indices = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.int64)
    values = np.array([5, 6, 7], dtype=np.int32)
    shape = np.array([2, 2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_tensor_3"}
    list_of_inputs.append(input_dict)

    # Input 4 (Empty SparseTensor)
    indices = np.empty((0, 2), dtype=np.int64)
    values = np.array([], dtype=np.float32)
    shape = np.array([3, 4], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_tensor_4"}
    list_of_inputs.append(input_dict)

    # Input 5 (Large SparseTensor)
    indices = np.array([[0, 9], [9, 0], [5, 5]], dtype=np.int64)
    values = np.array([10, 20, 30], dtype=np.float64)
    shape = np.array([10, 10], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_tensor_5"}
    list_of_inputs.append(input_dict)

    # Input 7 (Int64 values)
    indices = np.array([[1, 0], [0, 1], [0, 0]], dtype=np.int64)
    values = np.array([6, 5, 4], dtype=np.int64)
    shape = np.array([2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_tensor_7"}
    list_of_inputs.append(input_dict)

    # Input 8 (4D SparseTensor)
    indices = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=np.int64)
    values = np.array([8, 9, 10], dtype=np.int32)
    shape = np.array([1, 2, 1, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_tensor_8"}
    list_of_inputs.append(input_dict)

    # Input 9 (Unsorted indices)
    indices = np.array([[2,1],[0,3],[1,2],[0,0]],dtype=np.int64)
    values = np.array([1,2,3,4], dtype=np.int32)
    shape = np.array([3,4], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_tensor_9"}
    list_of_inputs.append(input_dict)

    # Input 10 (duplicate indices)
    indices = np.array([[0, 0], [0, 0], [1, 1], [1, 0]], dtype=np.int64)
    values = np.array([1, 5, 3, 4], dtype=np.int32)
    shape = np.array([2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    input_dict = {"sp_input": sp_input, "name": "reordered_tensor_10"}
    list_of_inputs.append(input_dict)


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.reorder"] = tf_sparse_reorder_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.reorder' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.reorder'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sparse.reorder', generated_inputs['tf.sparse.reorder'], lib="tf", suffix=0)
