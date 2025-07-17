
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_slice_inputs():
    list_of_inputs = []

    # Input 1
    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int32)
    shape = np.array([2, 5], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    start = np.array([0, 0], dtype=np.int64)
    size = np.array([2, 2], dtype=np.int64)
    name = "slice1"
    input_dict = {"sp_input": sp_input, "start": start, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = np.array([[0, 1], [1, 0], [1, 1]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.float32)
    shape = np.array([3, 3], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    start = np.array([0, 1], dtype=np.int64)
    size = np.array([2, 2], dtype=np.int64)
    name = "slice2"
    input_dict = {"sp_input": sp_input, "start": start, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    shape = np.array([2, 2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    start = np.array([0, 0, 0], dtype=np.int64)
    size = np.array([1, 2, 2], dtype=np.int64)
    name = "slice3"
    input_dict = {"sp_input": sp_input, "start": start, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    indices = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int64)
    shape = np.array([3, 3], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    start = np.array([1, 1], dtype=np.int64)
    size = np.array([1, 1], dtype=np.int64) # Reduced size to avoid out-of-bounds
    name = "slice4"
    input_dict = {"sp_input": sp_input, "start": start, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    indices = np.array([[0, 2], [1, 0]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.float64)
    shape = np.array([2, 5], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    start = np.array([0, 2], dtype=np.int64)
    size = np.array([1, 3], dtype=np.int64)
    name = "slice5"
    input_dict = {"sp_input": sp_input, "start": start, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    values = np.array([1, 2, 3, 4], dtype=np.int32)
    shape = np.array([2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    start = np.array([0, 0], dtype=np.int64)
    size = np.array([1, 1], dtype=np.int64)
    name = "slice6"
    input_dict = {"sp_input": sp_input, "start": start, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int32)
    shape = np.array([2, 5], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    start = np.array([0, 1], dtype=np.int64)
    size = np.array([1, 0], dtype=np.int64) #Corrected to [1,0]
    name = "slice7"
    input_dict = {"sp_input": sp_input, "start": start, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int32)
    shape = np.array([2, 5], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    start = np.array([0, 0], dtype=np.int64)
    size = np.array([0, 5], dtype=np.int64)
    name = "slice8"
    input_dict = {"sp_input": sp_input, "start": start, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int32)
    shape = np.array([2, 5], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    start = np.array([0, 0], dtype=np.int64)
    size = np.array([1, 2], dtype=np.int64)
    name = "slice9"
    input_dict = {"sp_input": sp_input, "start": start, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int32)
    shape = np.array([2, 5], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    start = np.array([1, 2], dtype=np.int64)
    size = np.array([1, 2], dtype=np.int64) #reduced size to avoid out-of-bounds
    name = "slice10"
    input_dict = {"sp_input": sp_input, "start": start, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.slice"] = tf_sparse_slice_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.slice' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.slice'.")

check_valid('tf.sparse.slice', generated_inputs['tf.sparse.slice'], lib="tf", suffix=0)
