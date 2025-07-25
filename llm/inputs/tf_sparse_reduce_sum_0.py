
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_sparse_reduce_sum_inputs():
    """
    Generates a list of valid inputs for the tf.sparse.reduce_sum function.

    The error "'tensorflow.python.framework.ops.EagerTensor' object has no
    attribute 'indices'" occurs because a dense tensor was passed to the API,
    which expects a sparse tensor. The fix is to construct and pass a
    `tf.SparseTensor` object for the `sp_input` parameter, as this object
    has the required '.indices' attribute that the API implementation uses.
    """
    list_of_inputs = []

    # Input 1: Basic 2D reduction along axis 0
    sp_input_1 = tf.sparse.SparseTensor(
        indices=np.array([[0, 1], [1, 2]], dtype=np.int64),
        values=np.array([1, 2], dtype=np.int32),
        dense_shape=np.array([2, 3], dtype=np.int64)
    )
    input_dict_1 = {
        'sp_input': sp_input_1,
        'axis': [0],
        'keepdims': False,
        'output_is_sparse': False,
        'name': 'reduce_axis_0'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D reduction along axis 1 with keepdims=True and float values
    sp_input_2 = tf.sparse.SparseTensor(
        indices=np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64),
        values=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        dense_shape=np.array([2, 3], dtype=np.int64)
    )
    input_dict_2 = {
        'sp_input': sp_input_2,
        'axis': [1],
        'keepdims': True,
        'output_is_sparse': False,
        'name': 'reduce_axis_1_keepdims'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Reduce all dimensions by providing a list of all axes
    input_dict_3 = {
        'sp_input': sp_input_2,
        'axis': [0, 1],
        'keepdims': False,
        'output_is_sparse': False,
        'name': 'reduce_all_axes'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Reduction with sparse output
    input_dict_4 = {
        'sp_input': sp_input_2,
        'axis': [0],
        'keepdims': False,
        'output_is_sparse': True,
        'name': 'reduce_sparse_output'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 3D tensor reduction along a negative axis
    sp_input_5 = tf.sparse.SparseTensor(
        indices=np.array([[0, 0, 0], [1, 1, 1], [1, 2, 0]], dtype=np.int64),
        values=np.array([10, 20, 30], dtype=np.int64),
        dense_shape=np.array([2, 3, 2], dtype=np.int64)
    )
    input_dict_5 = {
        'sp_input': sp_input_5,
        'axis': [-2],
        'keepdims': True,
        'output_is_sparse': False,
        'name': 'reduce_3d_negative_axis'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Empty sparse tensor
    sp_input_6 = tf.sparse.SparseTensor(
        indices=np.empty((0, 2), dtype=np.int64),
        values=np.array([], dtype=np.int32),
        dense_shape=np.array([5, 5], dtype=np.int64)
    )
    input_dict_6 = {
        'sp_input': sp_input_6,
        'axis': [0],
        'keepdims': False,
        'output_is_sparse': False,
        'name': 'reduce_empty_sparse'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Reduce all dimensions with an empty list for axis
    input_dict_7 = {
        'sp_input': sp_input_2,
        'axis': [],
        'keepdims': False,
        'output_is_sparse': False,
        'name': 'reduce_all_empty_axis'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    return list_of_inputs

generated_inputs["tf.sparse.reduce_sum"] = get_tf_sparse_reduce_sum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.reduce_sum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.reduce_sum'.")

check_valid('tf.sparse.reduce_sum', generated_inputs['tf.sparse.reduce_sum'], lib="tf", suffix=0)
