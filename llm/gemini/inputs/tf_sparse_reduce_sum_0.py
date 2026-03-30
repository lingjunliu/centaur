
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sparse_reduce_sum_inputs():
    list_of_inputs = []

    # Input 1
    indices = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64)
    values = np.array([1, 1, 1], dtype=np.int32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    axis = []
    keepdims = False
    output_is_sparse = False
    name = None
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64)
    values = np.array([1, 1, 1], dtype=np.int32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    axis = [0]
    keepdims = False
    output_is_sparse = False
    name = None
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64)
    values = np.array([1, 1, 1], dtype=np.int32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    axis = [1]
    keepdims = False
    output_is_sparse = False
    name = None
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    indices = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64)
    values = np.array([1, 1, 1], dtype=np.int32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    axis = [1]
    keepdims = True
    output_is_sparse = False
    name = None
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    indices = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64)
    values = np.array([1, 1, 1], dtype=np.int32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    axis = [0, 1]
    keepdims = False
    output_is_sparse = False
    name = None
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.float32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    axis = [0]
    keepdims = False
    output_is_sparse = True
    name = None
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    indices = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.int64)
    values = np.array([1, 2, 3, 4], dtype=np.int32)
    dense_shape = np.array([2, 2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    axis = [0]
    keepdims = False
    output_is_sparse = False
    name = None
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([-1, 2], dtype=np.int32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    axis = [1]
    keepdims = False
    output_is_sparse = False
    name = None
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([-1, 2], dtype=np.int32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    axis = [0]
    keepdims = True
    output_is_sparse = False
    name = None
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    indices = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    dense_shape = np.array([2, 2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    axis = [0, 1, 2]
    keepdims = False
    output_is_sparse = False
    name = None
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 - Negative axis
    indices = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64)
    values = np.array([1, 1, 1], dtype=np.int32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    axis = [-1]
    keepdims = False
    output_is_sparse = False
    name = None
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 - Empty SparseTensor
    indices = np.array([], dtype=np.int64).reshape(0, 2)
    values = np.array([], dtype=np.int32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    axis = [0]
    keepdims = False
    output_is_sparse = False
    name = None
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.reduce_sum"] = tf_sparse_reduce_sum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.reduce_sum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.reduce_sum'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sparse.reduce_sum', generated_inputs['tf.sparse.reduce_sum'], lib="tf", suffix=0)
