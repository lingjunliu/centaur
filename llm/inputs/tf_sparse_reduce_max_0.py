
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_reduce_max_inputs():
    list_of_inputs = []

    # Input 1
    indices = np.array([[0, 0], [0, 2], [1, 1]])
    values = np.array([1, 2, 3], dtype=np.int32)
    dense_shape = np.array([2, 3])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    axis = None
    keepdims = False
    output_is_sparse = False
    name = "reduce_max_1"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = np.array([[0, 0], [0, 2], [1, 1]])
    values = np.array([1, 2, 3], dtype=np.int32)
    dense_shape = np.array([2, 3])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    axis = [0]
    keepdims = False
    output_is_sparse = False
    name = "reduce_max_2"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = np.array([[0, 0], [0, 2], [1, 1]])
    values = np.array([1, 2, 3], dtype=np.int32)
    dense_shape = np.array([2, 3])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    axis = [1]
    keepdims = True
    output_is_sparse = False
    name = "reduce_max_3"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    indices = np.array([[0, 0], [1, 0], [1, 1]])
    values = np.array([-7, 4, 3], dtype=np.int32)
    dense_shape = np.array([3, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    axis = [1]
    keepdims = False
    output_is_sparse = False
    name = "reduce_max_4"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values = np.array([1, 2, 3, 4], dtype=np.int32)
    dense_shape = np.array([2, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    axis = [0, 1]
    keepdims = False
    output_is_sparse = False
    name = "reduce_max_5"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    indices = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
    values = np.array([1, 2, 3, 4], dtype=np.int32)
    dense_shape = np.array([2, 2, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    axis = [0]
    keepdims = True
    output_is_sparse = False
    name = "reduce_max_6"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    indices = np.array([[0, 0], [0, 2], [1, 1]])
    values = np.array([1, 2, 3], dtype=np.int32)
    dense_shape = np.array([2, 3])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    axis = [0]
    keepdims = False
    output_is_sparse = True
    name = "reduce_max_7"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    indices = np.array([[0, 0], [1, 1], [2, 2]])
    values = np.array([5, 2, 9], dtype=np.int32)
    dense_shape = np.array([3, 3])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    axis = [0]
    keepdims = True
    output_is_sparse = True
    name = "reduce_max_8"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    indices = np.array([[0, 0], [0, 2], [1, 1]])
    values = np.array([1, 2, 3], dtype=np.int32)
    dense_shape = np.array([2, 3])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    axis = [0,1]
    keepdims = True
    output_is_sparse = False
    name = "reduce_max_9"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values = np.array([1, 2, 3, 4], dtype=np.int32)
    dense_shape = np.array([2, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    axis = None
    keepdims = True
    output_is_sparse = False
    name = "reduce_max_10"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.reduce_max"] = tf_sparse_reduce_max_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.reduce_max' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.reduce_max'.")

check_valid('tf.sparse.reduce_max', generated_inputs['tf.sparse.reduce_max'], lib="tf", suffix=0)
