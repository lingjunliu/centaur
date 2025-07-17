
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_reduce_max_inputs():
    list_of_inputs = []

    # Input 1
    indices = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    axis = None
    keepdims = False
    output_is_sparse = False
    name = "sparse_reduce_max_1"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(input_dict)

    # Input 2
    indices = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, dense_shape)
    axis = [0]
    keepdims = False
    output_is_sparse = False
    name = "sparse_reduce_max_2"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(input_dict)

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
